from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

import trainer

class KNNClassifier:
    def __init__(self, details, verbose=False):
        self._details = details
        self._verbose = verbose

    def perform(self):
        neighbors = [10, 20, 30]
        learner = KNeighborsClassifier()
        pca = PCA(random_state=self._details.seed)

        params = {'pca__n_components': [15, 20],
                  'KNN__metric': ['manhattan', 'euclidean', 'chebyshev'], 'KNN__n_neighbors': neighbors,
                  'KNN__weights': ['uniform']}

        pipe = Pipeline([('pca', pca),
                         ('KNN', learner)])

        trainer.perform_experiment(ds=self._details.ds, ds_name=self._details.ds_name,
                                   clf_name='KNNClassifier', params=params,
                                   pipe=pipe,
                                   seed=self._details.seed)
