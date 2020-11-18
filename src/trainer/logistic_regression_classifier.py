from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from logger.logger import get_logger
import trainer

logger = get_logger()


class LRClassifier:
    def __init__(self, details, verbose=False):
        self._details = details
        self._verbose = verbose

    def perform(self):
        params = {'LRClassifier__penalty': ['l2']}

        learner = LogisticRegression(random_state=self._details.seed)

        pipe = Pipeline([
            ('LRClassifier', learner)])

        trainer.perform_experiment(ds=self._details.ds, ds_name=self._details.ds_name,
                                   clf_name='LRClassifier', params=params,
                                   pipe=pipe,
                                   seed=self._details.seed)
