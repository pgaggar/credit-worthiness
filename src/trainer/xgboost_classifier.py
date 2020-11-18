import xgboost as xgb
from sklearn.pipeline import Pipeline
import trainer


class XGBClassifier:
    def __init__(self, details, verbose=False):
        self._details = details
        self._verbose = verbose

    def perform(self):
        booster = xgb.XGBClassifier(booster='gbtree')

        params = {
            'Xgb__n_estimators': [10, 20, 50, 150],
            'Xgb__learning_rate': [0.02, 0.04],
            'Xgb__max_depth': [20, 40]
        }

        pipe = Pipeline([
            ('Xgb', booster)])

        trainer.perform_experiment(ds=self._details.ds, ds_name=self._details.ds_name,
                                   clf_name='Xgb', params=params, pipe=pipe,
                                   seed=self._details.seed)
