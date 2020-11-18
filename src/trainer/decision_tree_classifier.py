from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from logger.logger import get_logger
import trainer

logger = get_logger()


class DTClassifier:
    def __init__(self, details, verbose=False):
        self._details = details
        self._verbose = verbose

    def perform(self):
        params = {'DTClassifier__criterion': ['gini'],
                  'DTClassifier__max_depth': [20, 40, 60]}  # , 'DT__max_leaf_nodes': max_leaf_nodes}

        learner = DecisionTreeClassifier(random_state=self._details.seed)

        pipe = Pipeline([
            ('DTClassifier', learner)])

        trainer.perform_experiment(ds=self._details.ds, ds_name=self._details.ds_name,
                                   clf_name='DTClassifier', params=params,
                                   pipe=pipe,
                                   seed=self._details.seed)
