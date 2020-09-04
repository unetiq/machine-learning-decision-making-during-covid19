import numpy as np
import pandas as pd
import xgboost as xgb
import aws_access
from sklearn.model_selection import RepeatedStratifiedKFold


def load_data(data_loc):
    """
    Loads data for boosting models from AWS S3 and transforms it into pandas and xgboost.DMatrix format.

    :param data_loc: str; data location in S3
    :return (pandas.DataFrame, pandas.DataFrame, pandas.Series, xgboost.DMatrix); DataFrame with features and labels,
    DataFrame with features; Series with labels; xgboost.DMatrix with features and labels
    """

    data = pd.read_csv(aws_access.get_file_from_s3(data_loc))
    x = data.drop(["id", "y"], axis=1)
    y = data.y
    xgb_data = xgb.DMatrix(x, label=y)
    return data, x, y, xgb_data


def load_data_rnn(loc):
    """
    Loads data for the neural recurrent autoencoder given its AWS S3 location.

    :param loc: str; data location in S3
    :return np.array; data for the autoencoder
    """

    data = np.load(aws_access.get_file_from_s3(loc), allow_pickle=True)
    return data


def load_cv_folds(data, location):
    """
    Loads cross-validation folds data from AWS S3 and transforms it from list of IDs into list of indexes (required
    format for GridSearchCV and other tools).

    :param data: pandas.DataFrame; dataset with features and labels
    :param location: str; cross-validation folds location in S3 in a format of
    list(tuple(train_list(ID), test_list(ID)), ...)
    :return list; cross-validation folds in a format of list(tuple(train_list(indices), test_list(indices)), ...)
    """

    def map_fold_id_to_index(fold):
        id_index_map = dict(list(zip(data.id, data.index)))
        return list(map(lambda x: id_index_map[x], fold))

    cv_folds = np.load(aws_access.get_file_from_s3(location), allow_pickle=True)
    folds_processed = []
    for i in range(len(cv_folds)):
        folds_processed.append((
            map_fold_id_to_index(cv_folds[i][0]),
            map_fold_id_to_index(cv_folds[i][1])
        ))
    return folds_processed


def generate_cv_folds(data, n_splits, n_repeats, random_state):
    """
    Generates repeated stratified folds using RepeatedStratifiedKFold and given random_state.

    :param data: pandas.DataFrame; dataset with features and labels
    :param n_splits: int; number of cross-validation folds
    :param n_repeats: int; number of repetitions (# of different splits)
    :param random_state: int; random seed
    :return list; cross-validation folds in a format of list(tuple(train_list(indices), test_list(indices)), ...)
    """

    skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    folds = []
    for train, val in skf.split(data.id, data.y):
        folds.append((train, val))
    return folds
