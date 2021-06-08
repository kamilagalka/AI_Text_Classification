import logging
from time import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB  # noqa
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC  # noqa

from src import utils

logging.basicConfig(level=logging.DEBUG)

ALL_DATA_FILE_PATH = "../preprocessed_data/all_data_cleared.csv"


def train_and_evaluate(X: pd.DataFrame, y: pd.DataFrame, classifier, cross_validation: int = 10):
    logging.info(f'Starting evaluation for {classifier.__repr__()}')

    text_clf = Pipeline([
        ('vect', TfidfVectorizer()),
        ('clf', classifier),
    ])

    results = cross_val_score(text_clf, X, y, cv=cross_validation)
    logging.info(f"Cross validation scores: {results}")


def parameter_tuning(X: pd.DataFrame, y: pd.DataFrame, classifier, parameters: dict,
                     cross_validation: int = 10):
    logging.info(f'Starting parameters tuning for {classifier.__repr__()}')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    text_clf = Pipeline([
        ('vect', TfidfVectorizer()),
        ('clf', classifier),
    ])

    gs_clf = GridSearchCV(text_clf, parameters, cv=cross_validation, n_jobs=-1)
    gs_clf = gs_clf.fit(X_train, y_train)

    logging.info(f"Best params: {gs_clf.best_params_}")
    logging.info(f"Best score: {gs_clf.best_score_}")
    logging.info(f"Mean score: {gs_clf.cv_results_['mean_test_score']}")
    logging.info(f"Std score: {gs_clf.cv_results_['std_test_score']}")

    for param_name in sorted(parameters.keys()):
        logging.info("%s: %r" % (param_name, gs_clf.best_params_[param_name]))


if __name__ == '__main__':
    start = time()

    all_data: pd.DataFrame = pd.read_csv(ALL_DATA_FILE_PATH)
    # all_data = shuffle(all_data)
    # all_data = all_data[:1000]  # TODO: remove

    all_data = utils.clear_data(all_data, 'subj')

    nb_parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'vect__min_df': [1, 0.2, 0.4],
        'vect__max_df': [0.95, 0.7, 0.6],
        'clf__alpha': (1.0, 0.1, 0.01),
    }

    cls = MultinomialNB()  # 10_cv: 56%
    # cls = SVC(kernel='rbf')  # 10_cv: 71%
    # cls = SVC(kernel='linear')  # 10_cv: 61%

    train_and_evaluate(all_data['cleaned'], all_data['label'], cls, 10)
    # parameter_tuning(all_data['cleaned'], all_data['label'], cls, nb_parameters, 2)
    logging.info(f"Finished in {time() - start}")
