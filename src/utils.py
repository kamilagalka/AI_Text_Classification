import logging

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

logging.basicConfig(level=logging.DEBUG)

cachedStopWords = stopwords.words("english")


def clear_data(data: pd.DataFrame, cleaned_column_name: str) -> pd.DataFrame:
    logging.debug("Starting data clearing")
    stemmer = SnowballStemmer('english')
    words = stopwords.words('english')
    data['cleaned'] = data[cleaned_column_name].apply(
        lambda line: ' '.join([stemmer.stem(token) for token in line.split() if token not in words]))
    logging.debug("Finished data clearing")

    return data
