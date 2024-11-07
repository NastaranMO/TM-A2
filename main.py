from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
)


import pandas as pd
import evaluate
from utils import plot_distribution, align_labels_with_tokens
import numpy as np
from sklearn.metrics import classification_report as classification_report_sklearn
from seqeval.metrics import classification_report as classification_report_seqeval
from seqeval.metrics import f1_score, precision_score, recall_score

np.random.seed(42)
model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
metric = evaluate.load("seqeval")

ner_tags = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-LOC": 3,
    "I-LOC": 4,
    "B-ART": 5,
    "I-ART": 6,
    "B-MAT": 7,
    "I-MAT": 8,
    "B-SPE": 9,
    "I-SPE": 10,
    "B-CON": 11,
    "I-CON": 12,
}
number_to_ner_tag = {v: k for k, v in ner_tags.items()}
pos_tags = {
    "NN": 0,
    "CD": 1,
    ".": 2,
    "JJ": 3,
    "CC": 4,
    "DT": 5,
    "NNP": 6,
    "NNS": 7,
    "IN": 8,
    "VBN": 9,
    ",": 10,
    "(": 11,
    ")": 12,
    "VBG": 13,
    "VBD": 14,
    "RP": 15,
    "VBZ": 16,
    "TO": 17,
    "WDT": 18,
    "JJR": 19,
    "VB": 20,
    "POS": 21,
    "MD": 22,
    "RB": 23,
    "VBP": 24,
    "EX": 25,
    ":": 26,
    "NNPS": 27,
    "RBR": 28,
    "PRP": 29,
    "JJS": 30,
    "PRP$": 31,
    "``": 32,
    "''": 33,
    "WRB": 34,
    "RBS": 35,
    "LS": 36,
    "$": 37,
    "#": 38,
    "PDT": 39,
    "WP": 40,
    "FW": 41,
    "UH": 42,
    "SYM": 43,
    "WP$": 44,
}


def build_pos_tags_dict(ds):

    current_index = 0
    for line in ds["train"]:
        content = line["text"]
        if content.strip():
            _, pos, _ = content.split(maxsplit=2)
            if pos not in pos_tags:
                pos_tags[pos] = current_index
                current_index += 1

    return pos_tags


def process_ner(ner, sentence):
    sentence["ner_tags"].append(ner_tags[ner])


def process_pos(pos, sentence):
    sentence["pos_tags"].append(pos_tags[pos])


def generate_sentences(ds, ds_type):
    sentence = {"id": 0, "ner_tags": [], "tokens": [], "pos_tags": []}
    sentences = []
    # pos_tags = build_pos_tags_dict(ds)
    # print(f"pos tags for {ds_type}: {pos_tags}")
    for line in ds["train"]:
        content = line["text"]
        if content.strip() == "":
            sentences.append(sentence)
            sentence = {
                "id": len(sentences),
                "ner_tags": [],
                "tokens": [],
                "pos_tags": [],
            }
        else:
            word, pos, ner = content.split(maxsplit=2)
            sentence["tokens"].append(word)
            process_pos(pos, sentence)
            process_ner(ner, sentence)

    return sentences


def process_create_ds():
    # Load the raw data
    ds_train = load_dataset("text", data_files="./Data/train.txt")
    ds_test = load_dataset("text", data_files="./Data/test.txt")
    ds_val = load_dataset("text", data_files="./Data/val.txt")

    train_sentences = generate_sentences(ds_train, "train")
    test_sentences = generate_sentences(ds_test, "test")
    val_sentences = generate_sentences(ds_val, "validation")

    df_train = pd.DataFrame(train_sentences)
    df_val = pd.DataFrame(val_sentences)
    df_test = pd.DataFrame(test_sentences)

    return DatasetDict(
        {
            "train": Dataset.from_pandas(df_train),
            "validation": Dataset.from_pandas(df_val),
            "test": Dataset.from_pandas(df_test),
        }
    )


ds = process_create_ds()
plot_distribution(ds, ner_tags, number_to_ner_tag, verbose=False)
# print(f"ds traning: {ds['train'][0]}")
# print(f"pos tags: {pos_tags}")


# Part 3
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


tokenized_datasets = ds.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=ds["train"].column_names,
)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
batch = data_collator([tokenized_datasets["train"][i] for i in range(2)])


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    true_labels = [
        [number_to_ner_tag[l] for l in label if l != -100] for label in labels
    ]
    true_predictions = [
        [number_to_ner_tag[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    classification_details_sklearn = classification_report_sklearn(
        [label for sublist in true_labels for label in sublist],
        [pred for sublist in true_predictions for pred in sublist],
        output_dict=True,
    )

    report_cls_details_sklearn = pd.DataFrame(
        classification_details_sklearn
    ).transpose()

    classification_details_seqeval = classification_report_seqeval(
        true_labels, true_predictions, output_dict=True
    )

    report_cls_details_seqeval = pd.DataFrame(
        classification_details_seqeval
    ).transpose()

    report_cls_details_seqeval.to_csv("classification_report.csv")
    report_cls_details_sklearn.to_csv(
        "classification_report.csv", mode="a", header=False
    )

    f1_score_macro = f1_score(true_labels, true_predictions, average="macro")
    f1_score_micro = f1_score(true_labels, true_predictions, average="micro")

    return {
        "precision": precision_score(true_labels, true_predictions, zero_division=0),
        "recall": recall_score(true_labels, true_predictions, zero_division=0),
        "f1": f1_score(true_labels, true_predictions, zero_division=0),
        "f1_macro": f1_score_macro,
        "f1_micro": f1_score_micro,
        "accuracy": all_metrics["overall_accuracy"],
        "classification_report_details": classification_details_sklearn,
        "classfication_report_seqeval": classification_details_seqeval,
    }


id2label = {i: label for i, label in enumerate(number_to_ner_tag)}
label2id = {v: k for k, v in id2label.items()}


# Train the model
def train_model(checkpoint_name_saved, run=False):

    if run:
        print("Training the model")
        model = AutoModelForTokenClassification.from_pretrained(
            model_checkpoint,
            id2label=id2label,
            label2id=label2id,
        )
        # Default setting
        # args = TrainingArguments(
        #     output_dir=checkpoint_name_saved,
        #     do_train=True,
        #     do_eval=False,
        #     push_to_hub=True,
        # )

        args = TrainingArguments(
            output_dir=checkpoint_name_saved,
            do_train=True,
            do_eval=True,
            learning_rate=7.73381107021748e-05,
            weight_decay=0.01182365987726363,
            num_train_epochs=4,
            push_to_hub=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
        )
        trainer.train()
        trainer.push_to_hub(commit_message="Training complete")
        final_results = trainer.evaluate(tokenized_datasets["test"])
        print("Final Results:", final_results)
        # print(f"f1 score macro: {final_results['f1_macro']}")
        # print(f"f1 score micro: {final_results['f1_micro']}")


# Hyperparameter tuning
def hyper_tune(checkpoint_name_saved, model_checkpoint, run=False):
    if run:

        def optuna_hp_space(trial):
            return {
                "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 5e-4),
                "weight_decay": trial.suggest_loguniform("weight_decay", 1e-3, 0.1),
                "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 7),
            }

        def model_init():
            return AutoModelForTokenClassification.from_pretrained(
                model_checkpoint,
                id2label=id2label,
                label2id=label2id,
            )

        args = TrainingArguments(
            output_dir=checkpoint_name_saved,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            do_train=True,
            do_eval=True,
            push_to_hub=False,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
        )

        trainer = Trainer(
            model_init=model_init,
            args=args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
        )
        best_run = trainer.hyperparameter_search(
            direction="maximize",
            hp_space=optuna_hp_space,
            backend="optuna",
            n_trials=15,
        )
        print("Best Hyperparameters:", best_run.hyperparameters)
        print("Best Score:", best_run.objective)
        return best_run.hyperparameters


if __name__ == "__main__":
    # Part 3: Train the model with the default hyperparameters and after training evaluate the test set
    train_model("bert-finetuned-arcchialogy-ner-hp-tunned-hgf", run=True)
    # Part 4: Hyperparameter tuning
    best_hp = hyper_tune(
        "bert-finetuned-arcchialogy-ner-hp-tuning", model_checkpoint, run=False
    )


# hp best
# args = TrainingArguments(
#     output_dir=checkpoint_name_saved,
#     do_train=True,
#     do_eval=True,
#     learning_rate=7.73381107021748e-05,
#     weight_decay=0.01182365987726363,
#     num_train_epochs=4,
#     push_to_hub=True,
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
# )
