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

    def __init__(self, path=get_app_data_path(), verbose=False, seed=1):
        self._path = path
        self._verbose = verbose
        self.features = None
        self.output = None
        self.testing_x = None
        self.testing_y = None
        self.training_x = None
        self.training_y = None
        self._seed = seed
        self._data = pd.DataFrame()

    def build_train_test_split(self, test_size=0.3):
        if not self.training_x and not self.training_y and not self.testing_x and not self.testing_y:
            self.training_x, self.testing_x, self.training_y, self.testing_y = ms.train_test_split(
                self.features, self.output, test_size=test_size, random_state=self._seed
            )

    def load_and_process(self):
        """
        Load data from the given path and perform any initial processing required. This will populate the
        features and classes and is called before any processing is done.
        :return: Nothing
        """
        self._load_data()

        if self._verbose:
            logger.info("Processing {} Path: {}, Dimensions: {}".format(self.data_name(), self._path, self._data.shape))

        self._preprocess_data()

        self.get_features()
        self.get_output()
        logger.info("Feature dimensions: {}".format(self.features.shape))
        logger.info("Output dimensions: {}".format(self.output.shape))

    def get_features(self):
        logger.info("Pulling features")
        self.features = self._data.loc[:, self._data.columns != self.output_column_name()]

        return self.features

    def get_output(self):
        logger.info("Pulling output")
        self.output = np.array(self._data.loc[:, self._data.columns == self.output_column_name()])

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

    def _preprocess_data(self):
        full_df = self._data

        interval = (18, 25, 35, 60, 120)

        cats = ['Student', 'Young', 'Adult', 'Senior']
        full_df["Age_cat"] = pd.cut(full_df['Primary_applicant_age_in_years'], interval, labels=cats)

        full_df['Savings_account_balance'] = full_df['Savings_account_balance'].fillna('no_inf')
        full_df['Other_EMI_plans'] = full_df['Other_EMI_plans'].fillna('NA')

        full_df['Has_been_employed_for_at_least'] = full_df['Has_been_employed_for_at_least'].apply(
            lambda x: int(str(x).split()[0]) if str(x) != 'nan' else 0)

        full_df['Has_been_employed_for_at_most'] = full_df['Has_been_employed_for_at_most'].apply(
            lambda x: int(str(x).split()[0]) if str(x) != 'nan' else 0)

        full_df = full_df.merge(pd.get_dummies(full_df.Purpose, drop_first=True, prefix='Purpose'),
                                left_index=True, right_index=True)

        full_df = full_df.merge(pd.get_dummies(full_df.Gender, drop_first=True, prefix='Gender'), left_index=True,
                                right_index=True)

        full_df = full_df.merge(pd.get_dummies(full_df.Housing, drop_first=True, prefix='Housing'),
                                left_index=True, right_index=True)

        full_df = full_df.merge(
            pd.get_dummies(full_df.Savings_account_balance, drop_first=True, prefix='Savings'), left_index=True,
            right_index=True)

        full_df = full_df.merge(pd.get_dummies(full_df.Age_cat, drop_first=True, prefix='Age_cat'),
                                left_index=True, right_index=True)

        full_df = full_df.merge(pd.get_dummies(full_df.Marital_status, drop_first=True, prefix='Marital_status'),
                                left_index=True, right_index=True)

        full_df = full_df.merge(
            pd.get_dummies(full_df.Employment_status, drop_first=True, prefix='Employment_status'), left_index=True,
            right_index=True)

        full_df = full_df.merge(pd.get_dummies(full_df.Property, drop_first=True, prefix='Property'),
                                left_index=True, right_index=True)

        full_df = full_df.merge(pd.get_dummies(full_df.Loan_history, drop_first=True, prefix='Loan_history'),
                                left_index=True, right_index=True)

        full_df = full_df.merge(pd.get_dummies(full_df.Other_EMI_plans, drop_first=True, prefix='Other_EMI'),
                                left_index=True, right_index=True)

        del full_df["Purpose"]
        del full_df["Gender"]
        del full_df["Housing"]
        del full_df["Primary_applicant_age_in_years"]
        del full_df["Savings_account_balance"]
        del full_df["Age_cat"]
        del full_df["Marital_status"]
        del full_df["Employment_status"]
        del full_df["Telephone"]
        del full_df["loan_application_id"]
        del full_df["Property"]
        del full_df["Balance_in_existing_bank_account_(lower_limit_of_bucket)"]
        del full_df["Balance_in_existing_bank_account_(upper_limit_of_bucket)"]
        del full_df["Loan_history"]
        del full_df["Other_EMI_plans"]

        self._data = full_df
        print(self._data.columns)

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
