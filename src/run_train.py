import argparse
import pickle
from datetime import datetime
import time
import psutil
import numpy as np
import os
from utils.path_utils import get_app_data_path
from logger.logger import get_logger
from trainer import custom_accuracy
from trainer.decision_tree_classifier import DTClassifier
from trainer.random_forest_classifier import RFClassifier
from trainer.adaboost_classifier import AdaboostClassifier
from trainer.lightgbm_classifier import LightGBMClassifier
from trainer.xgboost_classifier import XGBClassifier
from trainer.logistic_regression_classifier import LRClassifier
from trainer.knn_classifier import KNNClassifier
from trainer.ann_classifier import ANNClassifier
from utils.load_and_process import DataLoader

logger = get_logger()


class TrainingDetails(object):
    def __init__(self, ds, ds_name, seed):
        self.ds = ds
        self.ds_name = ds_name
        self.seed = seed


def run_experiment(ds, experiment, timing_key, verbose, timings):
    """

    :param ds:
    :param experiment:
    :param timing_key:
    :param verbose:
    :param timings:
    :return:
    """
    data = ds['data']
    data.load_and_process()
    data.dump_test_train()
    experiment_details = TrainingDetails(
        data, ds['name'], seed=seed
    )

    t = datetime.now()
    exp = experiment(experiment_details, verbose=verbose)

    exp.perform()
    t_d = datetime.now() - t
    timings[timing_key] = t_d.seconds


def run_test_experiment(ds):
    """

    :param ds:
    :return:
    """
    start_time = time.process_time()
    data = ds['data']

    try:
        model_path = os.path.join(get_app_data_path(), 'output/model.pkl')
        with open(model_path, 'rb') as fin:
            model = pickle.load(fin)
        _, test_df = data.load_train_test()
        test_x = test_df.loc[:, test_df.columns != data.output_column_name()].values
        test_y = test_df.loc[:, test_df.columns == data.output_column_name()].values
        output = model.predict(test_x)
        y = test_y.ravel()
        acc = custom_accuracy(output, y)
        logger.info("Accuracy on test: %s", acc)

    except IOError:
        logger.info("Can't find file")
    end_time = time.process_time()
    logger.info("Total time for 100000 samples: %s seconds", end_time - start_time)
    process = psutil.Process(os.getpid())
    logger.info("Memory usage: %s MB", process.memory_info()[0] / (1024 * 1024))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform experiments')
    parser.add_argument('--threads', type=int, default=1, help='Number of threads (defaults to 1, -1 for auto)')
    parser.add_argument('--seed', type=int, help='A random seed to set, if desired')
    parser.add_argument('--boosting', action='store_true', help='Run the Boosting experiment')
    parser.add_argument('--rf', action='store_true', help='Run the Bagging experiment')
    parser.add_argument('--lgbm', action='store_true', help='Run the LightGBM experiment')
    parser.add_argument('--xgb', action='store_true', help='Run the XGBoost experiment')
    parser.add_argument('--boost', action='store_true', help='Run the Adaboost experiment')
    parser.add_argument('--dt', action='store_true', help='Run the Decision Tree experiment')
    parser.add_argument('--knn', action='store_true', help='Run the K Nearest Neighbors experiment')
    parser.add_argument('--ann', action='store_true', help='Run the ANN experiment')
    parser.add_argument('--lr', action='store_true', help='Run the Logistic Regression experiment')
    parser.add_argument('--test', action='store_true', help='Test the trained model')
    parser.add_argument('--verbose', action='store_true', help='If true, provide verbose output')
    args = parser.parse_args()
    verbose = args.verbose
    threads = args.threads

    seed = args.seed
    if seed is None:
        seed = np.random.randint(0, (2 ** 32) - 1)
        # print("Using seed {}".format(seed))

    # logger.info("Loading data")

    ds = {
        'data': DataLoader(verbose=verbose, seed=seed),
        'name': 'CreditRiskData',
        'readable_name': 'Credit Risk Data',
    }

    timings = {}

    if args.lgbm:
        run_experiment(ds, LightGBMClassifier, 'LGBClassifier', verbose, timings)

    if args.xgb:
        run_experiment(ds, XGBClassifier, 'XGBClassifier', verbose, timings)

    if args.boost:
        run_experiment(ds, AdaboostClassifier, 'AdaboostClassifier', verbose, timings)

    if args.rf:
        run_experiment(ds, RFClassifier, 'RFClassifier', verbose, timings)

    if args.lr:
        run_experiment(ds, LRClassifier, 'LRClassifier', verbose, timings)

    if args.dt:
        run_experiment(ds, DTClassifier, 'DTClassifier', verbose, timings)

    if args.knn:
        run_experiment(ds, KNNClassifier, 'KNNClassifier', verbose, timings)

    if args.ann:
        run_experiment(ds, ANNClassifier, 'ANNClassifier', verbose, timings)

    if args.test:
        run_test_experiment(ds)
