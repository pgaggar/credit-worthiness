import os
import pickle
from time import clock
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from logger.logger import get_logger
from utils.path_utils import get_app_data_path

from sklearn.model_selection import train_test_split, KFold, GridSearchCV

import matplotlib

matplotlib.use('Agg')

logger = get_logger()

OUTPUT_DIRECTORY = os.path.join(get_app_data_path(), 'output')

if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)

if not os.path.exists('{}/images'.format(OUTPUT_DIRECTORY)):
    os.makedirs('{}/images'.format(OUTPUT_DIRECTORY))


def custom_accuracy(truth, pred):
    diff = np.where(truth == pred, 1, 0)
    sm = np.sum(diff)

    return sm / len(truth)


scorer_accuracy = make_scorer(custom_accuracy)


def basic_results(learner, training_x, training_y, params, clf_name=None, dataset=None, seed=55, threads=1):
    """
    :param learner:
    :param training_x:
    :param training_y:
    :param params:
    :param clf_name:
    :param dataset:
    :param dataset_readable_name:
    :param seed:
    :param threads:
    :return:
    """

    logger.info("Computing basic results for {} ({} thread(s))".format(clf_name, threads))

    if clf_name is None or dataset is None:
        raise Exception('clf_type and dataset are required')
    if seed is not None:
        np.random.seed(seed)
    kfold = KFold(n_splits=10, random_state=seed)
    cv = GridSearchCV(learner, n_jobs=threads, param_grid=params, verbose=10, refit=True, cv=kfold,
                      scoring=scorer_accuracy)
    training_y = training_y.ravel()

    cv.fit(training_x, training_y)
    reg_table = pd.DataFrame(cv.cv_results_)
    reg_table.to_csv('{}/{}_{}_reg.csv'.format(OUTPUT_DIRECTORY, clf_name, dataset), index=False)

    best_estimator = cv.best_estimator_.fit(training_x, training_y)
    with open(os.path.join(OUTPUT_DIRECTORY, 'model.pkl'), 'wb') as fout:
        pickle.dump(best_estimator, fout)

    best_params = pd.DataFrame([best_estimator.get_params()])
    best_params.to_csv('{}/{}_{}_best_params.csv'.format(OUTPUT_DIRECTORY, clf_name, dataset), index=False)
    logger.info(" - Grid search complete")

    return cv


def make_timing_curve(x, y, clf, clf_name, dataset, dataset_readable_name, verbose=False, seed=42):
    """
    :param x:
    :param y:
    :param clf:
    :param clf_name:
    :param dataset:
    :param dataset_readable_name:
    :param verbose:
    :param seed:
    :return: None
    """
    logger.info("Building timing curve")

    # np.linspace(0.1, 1, num=10)  #
    sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    tests = 2
    out = dict()
    out['train'] = np.zeros(shape=(len(sizes), tests))
    out['test'] = np.zeros(shape=(len(sizes), tests))
    for i, frac in enumerate(sizes):
        for j in range(tests):
            np.random.seed(seed)
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 - frac, random_state=seed)
            st = clock()
            clf.fit(x_train, y_train)
            out['train'][i, j] = (clock() - st)
            st = clock()
            clf.predict(x_test)
            out['test'][i, j] = (clock() - st)
            logger.info(" - {} {} {}".format(clf_name, dataset, frac))

    train_df = pd.DataFrame(out['train'], index=sizes)
    test_df = pd.DataFrame(out['test'], index=sizes)
    plt = plot_model_timing('{} - {}'.format(clf_name, dataset_readable_name),
                            np.array(sizes) * 100, train_df, test_df)
    plt.savefig('{}/images/{}_{}_TC.png'.format(OUTPUT_DIRECTORY, clf_name, dataset), format='png', dpi=150)

    out = pd.DataFrame(index=sizes)
    out['train'] = np.mean(train_df, axis=1)
    out['test'] = np.mean(test_df, axis=1)
    out.to_csv('{}/{}_{}_timing.csv'.format(OUTPUT_DIRECTORY, clf_name, dataset))

    logger.info(" - Timing curve complete")


def perform_experiment(ds, ds_name, clf_name, params, pipe, seed=0, threads=1):
    """
    :param ds:
    :param ds_name:
    :param ds_readable_name:
    :param clf_name:
    :param params:
    :param pipe:
    :param seed:
    :param threads:
    :return: final_params
    """

    train_df, _ = ds.load_train_test()
    ds_training_x = train_df.loc[:, train_df.columns != ds.output_column_name()].values
    ds_training_y = train_df.loc[:, train_df.columns == ds.output_column_name()].values
    ds_clf = basic_results(pipe, ds_training_x, ds_training_y, params, clf_name, ds_name,
                           threads=threads, seed=seed)

    ds_final_params = ds_clf.best_params_
    pipe.set_params(**ds_final_params)

    # make_timing_curve(ds.features, ds.output, pipe, clf_name, ds_name, ds_readable_name,
    #                   seed=seed)

    return ds_final_params


def plot_model_timing(title, data_sizes, fit_scores, predict_scores, ylim=None):
    """
    :param title:
    :param data_sizes:
    :param fit_scores:
    :param predict_scores:
    :param ylim:
    :return: plot
    """
    plt.close()
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training Data Size (% of total)")
    plt.ylabel("Time (s)")
    fit_scores_mean = np.mean(fit_scores, axis=1)
    fit_scores_std = np.std(fit_scores, axis=1)
    predict_scores_mean = np.mean(predict_scores, axis=1)
    predict_scores_std = np.std(predict_scores, axis=1)
    plt.grid()
    plt.tight_layout()

    plt.fill_between(data_sizes, fit_scores_mean - fit_scores_std,
                     fit_scores_mean + fit_scores_std, alpha=0.2)
    plt.fill_between(data_sizes, predict_scores_mean - predict_scores_std,
                     predict_scores_mean + predict_scores_std, alpha=0.2)
    plt.plot(data_sizes, predict_scores_mean, 'o-', linewidth=1, markersize=4,
             label="Predict time")
    plt.plot(data_sizes, fit_scores_mean, 'o-', linewidth=1, markersize=4,
             label="Fit time")

    plt.legend(loc="best")
    return plt
