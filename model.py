import xgboost as xgb
import scipy
import copy
import numpy as np
import config
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV


def fit_grid_search_CV(x, y, grid_params, folds, fit_params={}):
    """
    Fine-tunes XGBoost model to the given data and folds using two-stage GridSearchCV.
    At the first stage performs broad search, and at the second stage - fine-grained search.

    :param x: object; data features
    :param y: object; data labels
    :param grid_params: dict; dictionary with parameters names (str) as keys and lists of parameter settings to try as values
    :param folds: list; a list containing (train, test) splits as arrays of indices
    :param fit_params: parameters passed to the fit method of the XGBClassifier
    :return: sklearn.model_selection.GridSearchCV object with the results of the grid search
    """

    for key in config.INITIAL_GRID_PARAMS:
        if key not in grid_params:
            grid_params[key] = config.INITIAL_GRID_PARAMS[key]

    grid_search = GridSearchCV(
        xgb.XGBClassifier(),
        grid_params,
        cv=folds,
        n_jobs=-1,
        scoring="roc_auc",
        verbose=2,
        refit=True)

    grid_search.fit(x.values, y, **fit_params)

    print("Best XGBoost parameters found with GridSearchCV:")
    print(grid_search.best_params_)

    print("\nMaximum XGBoost CV score (using the parameters above):")
    print(grid_search.best_score_)

    return grid_search


def generate_parameter_grid_for_2nd_iteration(prev_parameters):
    """
    Creates a new, more specific grid for GridSearchCV based on the parameters of the previously fitted estimator.

    :param prev_parameters: dict; dictionary with parameters names (str) as keys and values of previously fitted estimator
    :return: dict with parameters names (str) as keys and lists of parameter settings to try as values
    """

    c = prev_parameters["n_estimators"]
    n_estimators_subgrid = [c - 20, c, c + 20]
    if c - 20 <= 0: n_estimators_subgrid = n_estimators_subgrid[1:]

    c = prev_parameters["max_depth"]
    max_depth_subgrid = [c - 1, c, c + 1]
    if c - 1 == 1: max_depth_subgrid = max_depth_subgrid[1:]

    c = prev_parameters["min_child_weight"]
    min_child_weight_subgrid = [c - 1, c, c + 1]
    if c - 1 == 0: min_child_weight_subgrid = min_child_weight_subgrid[1:]

    c = prev_parameters["gamma"]
    gamma_subgrid = [c - 0.3, c, c + 0.3]
    if c - 0.3 < 0: gamma_subgrid = gamma_subgrid[1:]

    c = prev_parameters["subsample"]
    subsample_subgrid = [c - 0.05, c, c + 0.05]
    if c - 0.05 == 0: subsample_subgrid = subsample_subgrid[1:]

    c = prev_parameters["colsample_bytree"]
    colsample_bytree_subgrid = [c - 0.05, c, c + 0.05]
    if c - 0.05 == 0: colsample_bytree_subgrid = colsample_bytree_subgrid[1:]

    new_grid_params = {
        "n_estimators": n_estimators_subgrid,
        "max_depth": max_depth_subgrid,
        "min_child_weight": min_child_weight_subgrid,
        "gamma": gamma_subgrid,
        "subsample": subsample_subgrid,
        "colsample_bytree": colsample_bytree_subgrid,
    }

    return new_grid_params


def recursive_feature_selection(booster, x, y, folds, step, force_select_features=None, show_plot=True, plot_name=None):
    """
    Runs a recursive feature elimination (RFE) algorithm using provided estimator, prints and plots the results, and
    returns the selected features.

    :param booster: object; estimator object to use for feature elimination
    :param x: object; data features
    :param y: object; data labels
    :param folds: list; a list containing (train, test) splits as arrays of indices
    :param step: int; number of features to remove at each iteration of RFE
    :param force_select_features: list; if not None, list of features to forcefully include in the final subset
    :param show_plot: bool; to plot the features # vs CV score plot
    :param plot_name: str; if not None, name of the file to save plot to
    :return: (object, xgboost.DMatrix); tuple of data features and DMatrix containing only selected features
    """

    selector = RFECV(booster, step=step, min_features_to_select=1, cv=folds, scoring="roc_auc", verbose=2, n_jobs=-1)
    selector.fit(x.fillna(-1), y)

    if show_plot:
        plt.figure(figsize=(10, 5))
        plt.title("XGB CV score vs No of Features")
        plt.xlabel("Number of features selected")
        plt.ylabel("ROC AUC")
        plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
        if plot_name:
            plt.savefig(plot_name)
        plt.show()

    if force_select_features:
        selected_features = set(x.columns[selector.support_]).union(set(force_select_features))
        print("Selected {} features originally and have {} features after adding extra.".format(
            len(set(x.columns[selector.support_])), len(selected_features)))
    else:
        selected_features = set(x.columns[selector.support_])
        print("Selected {} features.".format(len(selected_features)))

    x_selected = x[selected_features]
    xgb_data_selected = xgb.DMatrix(x_selected, label=y)

    return x_selected, xgb_data_selected


def run_xgb_grid_search(xgb_data, x, y, cv_folds):
    """
    Fine-tunes XGBoost model to the given data and folds using two-stage GridSearchCV. At the first stage performs
    extensive search, and at the second stage - more specific search.

    :param xgb_data: xgboost.DMatrix; XGB Dataset
    :param x: object; input data features
    :param y: object; input data labels
    :param cv_folds: list; cross-validation folds to use
    :return: tuple with booster and best found params
             (fine-tuned XGBClassifier fitted on all data, best found XGB parameters)
    """

    print("\n[Finding optimal # trees]\n")
    xgb_params = copy.deepcopy(config.DEFAULT_XGB_PARAMS)
    history = xgb.cv(xgb_params, xgb_data, 1000, early_stopping_rounds=20, folds=cv_folds, metrics="auc")
    print(history)

    grid_params = copy.deepcopy(config.INITIAL_GRID_PARAMS)

    grid_params["n_estimators"] = [history.index[-1] - 20, history.index[-1], history.index[-1] + 20]
    if history.index[-1] - 20 < 0: grid_params["n_estimators"] = grid_params["n_estimators"][1:]

    print("\n[Grid search 1st stage]\n")
    grid_search = fit_grid_search_CV(x, y, grid_params, cv_folds)

    grid_params = generate_parameter_grid_for_2nd_iteration(grid_search.best_params_)
    print("\n[Grid search 2nd stage]\n")
    grid_search = fit_grid_search_CV(x, y, grid_params, cv_folds)
    xgb_params = grid_search.best_params_

    booster = xgb.XGBClassifier(**xgb_params)
    booster = booster.fit(x, y)

    return booster, xgb_params


def calc_mean_and_confidence_interval(l, confidence=0.95):
    """
    Helper function that calculates mean and confidence interval values for the given list.

    :param l: list; list of values
    :param confidence: float; confidence level
    :return: (float, float, float); lower endpoint of CI, mean, upper endpoint of CI
    """

    a = 1.0 * np.array(l)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def calc_xgb_cv_auc(params, x, y, cv_folds):
    """
    Calculates cross-validation score for XGBClassifier model using given parameters on given data and folds.

    :param params: dict; XGBClassifier parameters
    :param x: object; data features
    :param y: object; data labels
    :param cv_folds: list; cross-validation folds to use
    :return: (float, float, float); CI and mean for the score
    """

    scores = []
    for fold in cv_folds:
        train_ix, test_ix = fold

        booster = xgb.XGBClassifier(**params)
        booster = booster.fit(x[train_ix], y[train_ix])

        ypred = booster.predict_proba(x[test_ix])[:, 1]
        score = roc_auc_score(y[test_ix], ypred)
        scores.append(score)

    return calc_mean_and_confidence_interval(scores)


def calc_xgb_auc_based_on_multiple_runs_and_return_model(params, xtrain, ytrain, xtest, ytest):
    """
    Calculates test score and its CI for XGBClassifier model using given parameters on given data, and returns both
    score and the model.

    :param params: dict; XGBClassifier parameters
    :param xtrain: object; train data features
    :param ytrain: object; train data labels
    :param xtest: object; test data features
    :param ytest: object; test data labels
    :return: ((float, float, float), xgboost.XGBClassifier); CI and mean for the score and fitted model
    """

    scores = []

    for seed in range(config.N_CI_RUNS):
        booster = xgb.XGBClassifier(seed=seed, **params)
        booster = booster.fit(xtrain, ytrain)

        ypred = booster.predict_proba(xtest)[:, 1]
        score = roc_auc_score(ytest, ypred)
        scores.append(score)

    seed = np.argmax(scores)
    booster = xgb.XGBClassifier(seed=seed, **params)

    return calc_mean_and_confidence_interval(scores), booster


def calc_xgb_auc_based_on_multiple_runs(params, xtrain, ytrain, xtest, ytest):
    """
    Calculates test score and its CI for XGBClassifier model using given parameters on given data.
    See `calc_xgb_auc_based_on_multiple_runs_and_return_model` for parameters specification.

    :return: (float, float, float); CI and mean for the score
    """

    scores, model = calc_xgb_auc_based_on_multiple_runs_and_return_model(params, xtrain, ytrain, xtest, ytest)

    return scores


def fine_tune_and_calc_auc_based_on_multiple_runs(pretrained_model_loc, params, xtrain, ytrain, xtest, ytest):
    """
    Calculates test score and its CI for fine-tuned XGBClassifier model using given parameters on given data, and
    returns the scores.

    :param pretrained_model_loc: str; location of dumped pre-trained XGBClassifier model
    :param params: dict; XGBClassifier parameters
    :param xtrain: object; train data features
    :param ytrain: object; train data labels
    :param xtest: object; test data features
    :param ytest: object; test data labels
    :return: ((float, float, float), xgboost.XGBClassifier); CI and mean for the score and fitted model
    """

    scores = []

    for seed in range(config.N_CI_RUNS):
        booster = xgb.XGBClassifier(seed=seed, **params)
        booster = booster.fit(xtrain, ytrain, {"xgb_model": pretrained_model_loc})

        ypred = booster.predict_proba(xtest)[:, 1]
        score = roc_auc_score(ytest, ypred)
        scores.append(score)

    return calc_mean_and_confidence_interval(scores)
