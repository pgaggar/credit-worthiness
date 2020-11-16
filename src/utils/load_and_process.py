import pandas as pd
from logger.logger import get_logger
import numpy as np
from src.utils.path_utils import get_app_data_path
import os
from sklearn.pipeline import Pipeline
import sklearn.model_selection as ms
from sklearn.preprocessing import StandardScaler

logger = get_logger()


class DataLoader:

    def __init__(self, path, verbose=False, seed=1):
        self._path = path
        self._verbose = verbose
        self.features = None
        self.output = None
        self.testing_x = None
        self.testing_y = None
        self.training_x = None
        self.training_y = None
        self._data = pd.DataFrame()

    def load_and_process(self):
        """
        Load data from the given path and perform any initial processing required. This will populate the
        features and classes and is called before any processing is done.
        :return: Nothing
        """
        self._load_data()

        if self._verbose:
            logger.info("Processing {} Path: {}, Dimensions: {}".format(self.data_name(), self._path, self._data.shape))

        # self._preprocess_data()

        self.get_features()
        self.get_output()
        logger.info("Feature dimensions: {}".format(self.features.shape))
        logger.info("Output dimensions: {}".format(self.output.shape))

    def get_features(self):
        logger.info("Pulling features")
        self.features = self._data.iloc[:, 0:-1]

        return self.features

    def get_output(self):
        logger.info("Pulling output")
        self.output = np.array(self._data.iloc[:, -1])

        return self.output

    @staticmethod
    def output_column_name():
        return 'high_risk_applicant'

    @staticmethod
    def data_name():
        return 'CreditRiskData'

    def _load_data(self):
        app_csv = os.path.join(self._path, 'applicant.csv')
        loan_csv = os.path.join(self._path, 'loan.csv')

        applicant_df = pd.read_csv(app_csv, index_col='applicant_id')
        loan_df = pd.read_csv(loan_csv, index_col='applicant_id')

        self._data = applicant_df.join(loan_df)

    def dump_test_train_val(self, test_size=0.2, random_state=123):
        """
        :param test_size:
        :param random_state:
        :return: None
        """
        train_df, test_df = ms.train_test_split(self._data,
                                                test_size=test_size,
                                                random_state=random_state,
                                                stratify=self._data[self.output_column_name()])

        train_df, val_df = ms.train_test_split(train_df,
                                               test_size=test_size,
                                               random_state=random_state,
                                               stratify=train_df[self.output_column_name()])

        test_df.to_csv('{}/{}_test.csv'.format(get_app_data_path(), self.data_name()), index=False, header=False)
        train_df.to_csv('{}/{}_train.csv'.format(get_app_data_path(), self.data_name()), index=False, header=False)
        val_df.to_csv('{}/{}_validate.csv'.format(get_app_data_path(), self.data_name()), index=False, header=False)


if __name__ == '__main__':
    cd_data = DataLoader(path=get_app_data_path(), verbose=True)
    cd_data.load_and_process()
    feature_list = []
    for feature in cd_data.features:
        feature_list.append(feature)
    cd_data.dump_test_train_val()
    # logger.info(feature_list)
