import logging
import os

import pandas as pd

from src import utils

logging.basicConfig(level=logging.INFO)

DATA_DIR_PATH = "../scaledata"


def get_df_from_files(subdir_path):
    file_dict = {}
    for file in os.listdir(subdir_path):
        if not file.startswith('label.4class'):
            column_name = file.split('.')[0]
            file_path = os.path.join(subdir_path, file)
            with open(file_path) as file:
                data = file.read().splitlines()

            file_dict[column_name] = data

    result = pd.DataFrame(data=file_dict)
    return result


def read_all_data():
    logging.info("Reading all data")
    all_data_df = pd.DataFrame()

    for directory in os.listdir(DATA_DIR_PATH):
        subdir_path = os.path.join(DATA_DIR_PATH, directory)
        author_df = get_df_from_files(subdir_path)
        all_data_df = pd.concat([all_data_df, author_df])

    all_data_df.to_csv('all_data.csv', index=False)


def read_all_data_remove_stop_words():
    logging.info("Reading all data, removing stop words")
    all_data_df = pd.DataFrame()

    for directory in os.listdir(DATA_DIR_PATH):
        subdir_path = os.path.join(DATA_DIR_PATH, directory)
        author_df = get_df_from_files(subdir_path)
        all_data_df = pd.concat([all_data_df, author_df])

    no_stop_words = []
    for text in all_data_df['subj']:
        no_stop_words.append(utils.remove_stop_words(text))

    all_data_df['subj_no_stop_words'] = no_stop_words

    all_data_df.to_csv('all_data_no_stop_words.csv', index=False)


if __name__ == '__main__':
    # read_all_data()
    read_all_data_remove_stop_words()
