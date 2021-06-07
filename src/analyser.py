import logging

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import src.utils as utils

logging.basicConfig(level=logging.INFO)

ALL_DATA_FILE_PATH = "../preprocessed_data/all_data_cleared.csv"


def classes_occurance_frequency(data_df: pd.DataFrame, column_name: str) -> dict[int, int]:
    return data_df[column_name].value_counts().to_dict()


def classes_avg_length(data_df: pd.DataFrame, group_by_column_name: str, checked_column_name: str) -> dict[int, float]:
    return (data_df.groupby(group_by_column_name)[checked_column_name].apply(lambda x: np.mean(x.str.len()))).to_dict()


def classes_most_popular_words(words_count: dict[int, [dict[str, int]]], num_of_words: int) -> dict[
    int, dict[str, int]]:
    res = {}
    for group in words_count:
        res[group] = {k: words_count[group][k] for k in list(words_count[group])[:num_of_words]}

    return res


def classes_least_popular_words(words_count: dict[int, [dict[str, int]]], num_of_words: int) -> dict[
    int, dict[str, int]]:
    res = {}
    for group in words_count:
        res[group] = {k: words_count[group][k] for k in list(words_count[group])[-num_of_words:]}

    return res


def _get_classes_words_count(data_df: pd.DataFrame, group_by_column_name: str, checked_column_name: str, ) -> dict[
    int, dict[str, int]]:
    data_df.assign(words=data_df[checked_column_name].str.lower().str.split()).explode('words').groupby(
        group_by_column_name)['words'].value_counts()

    data_df['words'] = data_df[checked_column_name].str.lower().str.split()
    df = data_df.explode('words').groupby(group_by_column_name)['words'].value_counts()

    res = dict(zip([group for group in data_df[group_by_column_name]], {}))

    for group in data_df[group_by_column_name].unique():
        res[group] = df[group].to_dict()
        res[group] = {word: count for word, count in res[group].items() if len(word) > 2}
    return res


if __name__ == '__main__':
    all_data: pd.DataFrame = pd.read_csv(ALL_DATA_FILE_PATH)

    occurances = classes_occurance_frequency(all_data, 'label')
    logging.info(f"Classes occurances in available data: {occurances}")

    avg_length = classes_avg_length(all_data, 'label', 'subj')
    logging.info(f"Classes avg length in available data: {avg_length}")

    # no_stop_words = []
    # for text in all_data['subj']:
    #     no_stop_words.append(utils.remove_stop_words(text))
    #
    # all_data['cleaned'] = no_stop_words

    classes_word_counts = _get_classes_words_count(all_data, 'label', 'cleaned')

    most_popular_words = classes_most_popular_words(classes_word_counts, num_of_words=20)
    logging.info(f"Classes most popular words in available data: {most_popular_words}")

    least_popular_words = classes_least_popular_words(classes_word_counts, num_of_words=10)
    logging.info(f"Classes least popular words in available data: {least_popular_words}")

    for label, words_count in most_popular_words.items():
        plt.title(f"Class {label} - most popular words")
        plt.bar(*zip(*words_count.items()))
        plt.xticks(rotation=45)
        plt.show()
    #
    # for label, words_count in least_popular_words.items():
    #     plt.title(f"Class {label} - least popular words")
    #     plt.bar(*zip(*words_count.items()))
    #     plt.xticks(rotation=45)
    #     plt.show()
