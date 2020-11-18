import lightgbm as lgb
import xgboost as xgb
from sklearn.pipeline import Pipeline
import trainer


class GBClassifier:
    def __init__(self, details, verbose=False):
        self._details = details
        self._verbose = verbose

    def perform(self):
        booster = lgb.LGBMClassifier(boosting_type='gbdt', class_weight=None,
                                     colsample_bytree=0.9234,
                                     metric='None', min_child_samples=399, min_child_weight=0.1,
                                     min_split_gain=0.0, n_jobs=4,
                                     objective=None, random_state=314, reg_alpha=2, reg_lambda=5,
                                     silent=True, subsample=0.855, subsample_for_bin=200000,
                                     subsample_freq=1)

        params = {
            'Boost__n_estimators': [10, 20, 50, 150],
            'Boost__learning_rate': [0.02, 0.04],
            'Boost__max_depth': [20, 40],
            'Boost__num_leaves': [30, 40, 50]
        }

        pipe = Pipeline([
            ('Boost', booster)])

        trainer.perform_experiment(ds=self._details.ds, ds_name=self._details.ds_name,
                                   clf_name='Boost', params=params, pipe=pipe,
                                   seed=self._details.seed)
