import pandas as pd
from logger.logger import get_logger
import numpy as np
from src.utils.path_utils import get_app_root_path
import os

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

if __name__ == '__main__':
    data_path = os.path.join(get_app_root_path(), 'data')
    cd_data = DataLoader(path=data_path, verbose=True)
    cd_data.load_and_process()
    feature_list = []
    for feature in cd_data.features:
        feature_list.append(feature)
    # logger.info(feature_list)
