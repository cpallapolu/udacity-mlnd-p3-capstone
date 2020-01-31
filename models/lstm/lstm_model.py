
import pandas as pd
import numpy as np

from os import path
from sklearn.metrics import mean_squared_error

from keras.layers import Dense, Dropout, LSTM, Bidirectional
from keras.models import Sequential


class LSTMPredictor:
    def __init__(self):
        self.data_dir = '../../datasets'

        if not path.exists(self.data_dir):
            raise Exception(
                '{} directory not found.'.format(self.data_dir)
            )

        self.train_file = '{}/{}'.format(self.data_dir, 'train.zip')
        self.val_file = '{}/{}'.format(self.data_dir, 'val.zip')
        self.pred_val_file = '{}/{}'.format(
            self.data_dir, 'lstm_pred_val.zip'
        )
        self.test_file = '{}/{}'.format(self.data_dir, 'test.zip')
        self.pred_test_file = '{}/{}'.format(
            self.data_dir, 'lstm_pred_test.zip'
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

        self.shaped_train_X = self.train_X.reshape(
            (self.train_X.shape[0], 1, self.train_X.shape[1])
        )
        self.shaped_val_X = self.val_X.reshape(
            (self.val_X.shape[0], 1, self.val_X.shape[1])
        )
        self.shaped_test_X = self.test_X.reshape(
            (self.test_X.shape[0], 1, self.test_X.shape[1])
        )

        print('\nShape of the train dataset: {}'.format(self.train_X.shape))
        print('\nShape of the val dataset: {}'.format(self.val_X.shape))
        print('\nShape of the test dataset: {}\n'.format(self.test_X.shape))

        print('\nShape of the reshaped train dataset: {}'.format(
            self.shaped_train_X.shape)
        )
        print('\nShape of the reshaped val dataset: {}'.format(
            self.shaped_val_X.shape)
        )
        print('\nShape of the reshaped test dataset: {}\n'.format(
            self.teshaped_test_Xst_X.shape)
        )

    def lstm_model(self):
        model = Sequential()

        model.add(
            Bidirectional(
                LSTM(
                    256,
                    recurrent_dropout=0.2,
                    kernel_initializer='lecun_normal',
                    return_sequences=True
                )
            )
        )
        model.add(
            Bidirectional(
                LSTM(
                    128,
                    recurrent_dropout=0.2,
                    kernel_initializer='lecun_normal'
                )
            )
        )
        model.add(Dense(50, activation='sigmoid'))
        model.add(Dropout(0.1))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(1, activation='linear'))

        model.compile(optimizer='adam', loss='mse')

        model.fit(
            self.shaped_train_X,
            self.train_log_y,
            epochs=5,
            batch_size=64,
            validation_data=(self.shaped_val_X, self.val_log_y),
            validation_freq=2,
            verbose=1,
            shuffle=False
        )

        self.model = model

    def lstm_train(self):
        self.lgbm_model()

    def lstm_predict(self, X):
        return self.model.predict(X)

    def predict(self):
        self.prev_val = self.lstm_predict(self.shaped_val_X)
        self.prev_test = self.lstm_predict(self.shaped_test_X)

    def evaluate_val_prediction(self):
        pred_val = self.pred_val.reshape(-1)

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
        pred_test = self.pred_test.reshape(-1)

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
