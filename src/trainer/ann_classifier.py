import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

import trainer


class ANNClassifier:
    def __init__(self, details, verbose=False):
        self._details = details
        self._verbose = verbose

    def perform(self):
        # Search for good alphas
        alphas = [10 ** -x for x in np.arange(-1, 9.01, 1.0)]

        hiddens = [(50,), (50, 100, 50)]
        learning_rates = sorted([0.001, 0.005])

        params = {'MLP__activation': ['relu'], 'MLP__alpha': alphas,
                  'MLP__learning_rate_init': learning_rates,
                  'MLP__hidden_layer_sizes': hiddens}

        learner = MLPClassifier(max_iter=1000, early_stopping=True, random_state=self._details.seed, shuffle=True,
                               verbose=self._verbose)

        pipe = Pipeline([('MLP', learner)])

        trainer.perform_experiment(ds=self._details.ds, ds_name=self._details.ds_name,
                                   clf_name='ANNClassifier', params=params,
                                   pipe=pipe,
                                   seed=self._details.seed)