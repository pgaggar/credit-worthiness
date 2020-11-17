import argparse
import pickle
from datetime import datetime
from time import clock

import numpy as np
from sklearn.metrics import mean_squared_error

from logger.logger import get_logger
from trainer import custom_accuracy
from trainer.decision_tree_classifier import DTClassifier
from utils.load_and_process import DataLoader

logger = get_logger()


class TrainingDetails(object):
    def __init__(self, ds, ds_name, ds_readable_name, seed):
        self.ds = ds
        self.ds_name = ds_name
        self.ds_readable_name = ds_readable_name
        self.seed = seed


def run_experiment(ds, experiment, timing_key, verbose, timings):
    """
    Run the training experiment chosen by the user
    :param details:
    :param experiment:
    :param timing_key:
    :param verbose:
    :param timings:
    :return: None
    """
    data = ds['data']
    data.load_and_process()
    data.build_train_test_split()
    experiment_details = TrainingDetails(
        data, ds['name'], ds['readable_name'],
        seed=seed
    )

    t = datetime.now()
    exp = experiment(experiment_details, verbose=verbose)

    logger.info("Running {} experiment: {}".format(timing_key, experiment_details.ds_readable_name))
    exp.perform()
    t_d = datetime.now() - t
    timings[timing_key] = t_d.seconds


def run_test_experiment(ds):
    """
    Run the testing experiment on the hold-out dataset
    :param details:
    :return:
    """
    start_time = clock()
    data = ds['data']
    data.load_and_impute()
    data.build_train_test_split()
    experiment_details = TrainingDetails(
        data, ds['name'], ds['readable_name'],
        seed=seed
    )
    try:
        with open('./output/model.pkl', 'rb') as fin:
            model = pickle.load(fin)
        data = experiment_details.ds.get_features()
        output = model.predict(data)
        y = experiment_details.ds.get_output()
        mse = np.sqrt(mean_squared_error(output, y))
        acc = np.sqrt(custom_accuracy(output, y))
        logger.info("RMSE: %s", mse)
        logger.info("Accuracy: %s", acc)
        output = output.astype(str)
        output_str = "\n".join(output)
        with open('output/model_output.txt', 'wb+') as fout:
            fout.write(output_str.encode('utf8'))


    except IOError:
        logger.info("Can't find file")
    end_time = clock()
    # logger.info("Total time for 100000 samples: %s seconds", end_time - start_time)
    # process = psutil.Process(os.getpid())
    # logger.info("Memory usage: %s MB", process.memory_info()[0] / (1024 * 1024))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform experiments')
    parser.add_argument('--threads', type=int, default=1, help='Number of threads (defaults to 1, -1 for auto)')
    parser.add_argument('--seed', type=int, help='A random seed to set, if desired')
    parser.add_argument('--boosting', action='store_true', help='Run the Boosting experiment')
    parser.add_argument('--bagging', action='store_true', help='Run the Bagging experiment')
    parser.add_argument('--gdboosting', action='store_true', help='Run the Gradient Boosting experiment')
    parser.add_argument('--knn', action='store_true', help='Run the KNN experiment')
    parser.add_argument('--dtclf', action='store_true', help='Run the Decision Tree experiment')
    parser.add_argument('--ann', action='store_true', help='Run the ANN experiment')
    parser.add_argument('--averaged', action='store_true', help='Run the Averaged experiment')
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

    # if args.boosting:
    #     run_experiment(ds, experiments.BoostingExperiment, 'Boosting', verbose, timings)
    #
    # if args.gdboosting:
    #     run_experiment(ds, experiments.GradientBoostingExperiment, 'GradientBoosting', verbose, timings)
    #
    # if args.bagging:
    #     run_experiment(ds, experiments.BaggingExperiment, 'Bagging', verbose, timings)
    #
    # if args.ann:
    #     run_experiment(ds, experiments.ANNExperiment, 'ANN', verbose, timings)
    #
    # if args.averaged:
    #     run_experiment(ds, experiments.AveragedExperiment, 'Averaged', verbose, timings)
    #
    # if args.knn:
    #     run_experiment(ds, experiments.KNNExperiment, 'KNN', verbose, timings)

    if args.dtclf:
        run_experiment(ds, DTClassifier, 'DTClassifier', verbose, timings)

    if args.test:
        run_test_experiment(ds)