from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt


def count_tags(ds, number_to_ner_tag):
    tags = []
    for item in ds:
        tags.extend(number_to_ner_tag[i] for i in item["ner_tags"])
    return Counter(tags)


def calculate_percentages(counts):
    total = sum(counts.values())
    return {tag: (count / total) * 100 for tag, count in counts.items()}


def analyze_ds(datasets, ner_tags, number_to_ner_tag, verbose=False):
    print(f"Named entity tags: {ner_tags}")
    if verbose:
        distributions = {}
        for set_name, dataset in datasets.items():
            tag_counts = count_tags(dataset, number_to_ner_tag)
            percentages = calculate_percentages(tag_counts)
            distributions[set_name] = percentages

        dist_df = pd.DataFrame(distributions).fillna(0)
        return dist_df


def plot_distribution(ds, ner_tags, number_to_ner_tag, verbose=False):
    if verbose:
        dist_df = analyze_ds(ds, ner_tags, number_to_ner_tag, verbose=verbose)
        dist_df.plot(kind="bar", width=0.8)
        plt.title(
            "Distribution of Named Entity Tag Across Train, Validation, and Test Sets"
        )
        plt.xlabel("Named Entity Tag")
        plt.ylabel("Percentage")
        plt.xticks(rotation=45)
        plt.legend(title="Dataset")
        plt.savefig("ner_distribution.png")


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels
