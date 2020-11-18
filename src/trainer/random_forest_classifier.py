from sklearn.pipeline import Pipeline
from sklearn import ensemble
import trainer


class RFClassifier:
    def __init__(self, details, verbose=False):
        self._details = details
        self._verbose = verbose

    def perform(self):
        random_forest = ensemble.RandomForestClassifier(
            random_state=self._details.seed)

        params = {"RF__max_depth": [3, 5, 7, 10, None],
                  "RF__n_estimators": [3, 5, 10, 25, 50, 150],
                  "RF__max_features": [4, 7, 15, 20]}

        pipe = Pipeline([
            ('RF', random_forest)])

        trainer.perform_experiment(ds=self._details.ds, ds_name=self._details.ds_name,
                                   clf_name='RF', params=params, pipe=pipe,
                                   seed=self._details.seed)
