
import pandas as pd
import numpy as np

from lightgbm import LGBMModel
from os import path
from sklearn.metrics import mean_squared_error


class LightGBM:
    def __init__(self):
        self.data_dir = '../../datasets'

        if not path.exists(self.data_dir):
            raise Exception(
                '{} directory not found.'.format(self.data_dir)
            )

        self.train_file = '{}/{}'.format(self.data_dir, 'train.zip')
        self.val_file = '{}/{}'.format(self.data_dir, 'val.zip')
        self.pred_val_file = '{}/{}'.format(
            self.data_dir, 'lgbm_pred_val.zip'
        )
        self.test_file = '{}/{}'.format(self.data_dir, 'test.zip')
        self.pred_test_file = '{}/{}'.format(
            self.data_dir, 'lgbm_pred_test.zip'
        )

    def load_data(self, zip_path):
        df = pd.read_csv(
            zip_path,
            dtype={'fullVisitorId': 'str'},
            compression='zip'
        )

        [rows, columns] = df.shape

        print('\nLoaded {} rows with {} columns from {}.\n'.format(
            rows, columns, zip_path
        ))

        return df

    def load(self):
        print('Loading train data from {}'.format(self.train_file))
        self.train_df = self.load_data(self.train_file)

        print('Loading val data from {}'.format(self.val_file))
        self.val_df = self.load_data(self.val_file)

        print('Loading test data from {}'.format(self.test_file))
        self.test_df = self.load_data(self.test_file)

    def prepare_data(self):
        train_df = self.train_df
        val_df = self.val_df
        test_df = self.test_df

        self.train_id = train_df['fullVisitorId'].values
        self.val_id = val_df['fullVisitorId'].values
        self.test_id = test_df['fullVisitorId'].values

        self.train_y = train_df['totals.transactionRevenue'].values
        self.train_log_y = np.log1p(self.train_y)

        self.val_y = val_df['totals.transactionRevenue'].values
        self.val_log_y = np.log1p(self.val_y)

        self.train_X = train_df.drop(
            ['totals.transactionRevenue', 'fullVisitorId'],
            axis=1
        )
        self.val_X = val_df.drop(
            ['totals.transactionRevenue', 'fullVisitorId'],
            axis=1
        )
        self.test_X = test_df.drop(['fullVisitorId'], axis=1)

        print('\nShape of the train dataset: {}'.format(self.train_X.shape))
        print('\nShape of the val dataset: {}'.format(self.val_X.shape))
        print('\nShape of the test dataset: {}\n'.format(self.test_X.shape))

    def lgbm_model(self):
        self.model = LGBMModel(
            objective='regression',
            metric='rmse',
            n_estimators=1000,
            learning_rate=0.01,
            min_child_samples=100,
            bagging_fraction=0.7,
            feature_fraction=0.5,
            bagging_freq=5,
            bagging_seed=2020
        )

        self.model = self.model.fit(
            self.train_X,
            self.train_log_y,
            eval_set=(self.val_X, self.val_log_y),
            early_stopping_rounds=100,
            verbose=100
        )

    def lgbm_predict(self, X):
        return self.model.predict(X, self.model.best_iteration_)

    def lgbm_train(self):
        self.lgbm_model()

    def predict(self):
        self.prev_val = self.lgbm_predict(self.val_X)
        self.prev_test = self.lgbm_predict(self.test_X)

    def evaluate_val_prediction(self):
        pred_val = self.prev_val

        pred_val[pred_val < 0] = 0
        pred_val_data = {
            'fullVisitorId': self.val_id,
            'transactionRevenue': self.val_y,
            'predictedRevenue': np.expm1(pred_val)
        }

        pred_val_df = pd.DataFrame(pred_val_data)

        pred_val_df = pred_val_df.groupby('fullVisitorId')
        pred_val_df = pred_val_df['transactionRevenue', 'predictedRevenue']\
            .sum().reset_index()

        rsme_val = np.sqrt(
            mean_squared_error(
                np.log1p(pred_val_df['transactionRevenue'].values),
                np.log1p(pred_val_df['predictedRevenue'].values)
            )
        )

        self.rsme_val = rsme_val
        self.prev_val_df = pred_val_df

    def evaluate_test_prediction(self):
        pred_test = self.pred_test

        pred_test[pred_test < 0] = 0

        pred_test_data = {
            'fullVisitorId': self.test_id,
            'predictedRevenue': np.expm1(pred_test)
        }

        pred_test_df = pd.DataFrame(pred_test_data)

        pred_test_df = pred_test_df.groupby('fullVisitorId')
        pred_test_df = pred_test_df['predictedRevenue'].sum().reset_index()

        self.pred_test_df = pred_test_df

    def write_to_csv(self):
        self.pred_val_df.to_csv(
            self.pred_val_file,
            index=False,
            compression='zip'
        )

        self.pred_test_df.to_csv(
            self.pred_test_file,
            index=False,
            compression='zip'
        )
