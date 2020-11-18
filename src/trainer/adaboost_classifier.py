from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import trainer


class AdaboostClassifier:
    def __init__(self, details, verbose=False):
        self._details = details
        self._verbose = verbose

    def perform(self):
        booster = GradientBoostingClassifier(random_state=self._details.seed)

        params = {
            'Boost__n_estimators': [10, 20, 50, 150],
            'Boost__learning_rate': [0.02, 0.04],
            'Boost__max_depth': [20, 40]
        }

        pipe = Pipeline([
            ('Boost', booster)])

        trainer.perform_experiment(ds=self._details.ds, ds_name=self._details.ds_name,
                                   clf_name='Boost', params=params, pipe=pipe,
                                   seed=self._details.seed)
