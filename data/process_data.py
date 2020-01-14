
import json
import pandas as pd
from os import path, makedirs

from datetime import datetime
from dateutil.parser import parse
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from pandas.io.json import json_normalize


class ProcessData:
    def __init__(self):
        self.originals_dir = '../data/originals'
        self.files_dir = '../data/files'

        self.input_train_file = '{}/{}'.format(self.originals_dir, 'train.zip')
        self.input_test_file = '{}/{}'.format(self.originals_dir, 'test.zip')

        self.train_file = '{}/{}'.format(self.files_dir, 'train.zip')
        self.val_file = '{}/{}'.format(self.files_dir, 'val.zip')
        self.test_file = '{}/{}'.format(self.files_dir, 'test.zip')

    def load_data(self, data_path):
        JSON_COLUMNS = [
            'device', 'geoNetwork', 'totals', 'trafficSource'
        ]

        df = pd.read_csv(
            data_path,
            converters={column: json.loads for column in JSON_COLUMNS},
            dtype={'fullVisitorId': 'str'},
            compression='zip'
        )

        for column in JSON_COLUMNS:
            column_as_df = json_normalize(df[column])
            column_as_df.columns = [
                f'{column}.{subcolumn}'
                for subcolumn in column_as_df.columns
            ]
            df = df.drop(column, axis=1)
            df = df.merge(column_as_df, right_index=True, left_index=True)

        [rows, columns] = df.shape

        print(
            '\nLoaded {} rows with {} columns from {}.\n'.format(
                rows, columns, data_path
            )
        )

    def load(self):
        if not path.exists(self.originals_dir):
            raise Exception('{} directory not found.'.format(
                self.originals_dir
            ))

        print('Loading train data from {}'.format(self.input_train_file))
        self.train_df = self.load_data(self.input_train_file)

        print('Loading test data from {}'.format(self.input_test_file))
        self.test_df = self.load_data(self.input_test_file)

    def missing_data(self):
        train_df = self.train_df
        visitor_id = 'fullVisitorId'
        revenue_id = 'totals.transactionRevenue'

        total_null_sum = train_df.isnull().sum()
        total_null_count = train_df.isnull().count()

        train_df[revenue_id] = train_df[revenue_id].astype('float')

        dfgv_id = train_df.groupby(visitor_id)
        dfgv_id = dfgv_id[revenue_id].sum().reset_index()

        total = total_null_sum.sort_values(ascending=False)
        percent = (total_null_sum / total_null_count * 100)
        percent = percent.sort_values(ascending=False)

        df = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        rows = train_df.shape[0]
        uniq_rows = dfgv_id.shape[0]

        uniq_visitors = train_df[visitor_id].nunique()

        nzi = pd.notnull(train_df["totals.transactionRevenue"]).sum()
        nzr = (dfgv_id[revenue_id] > 0).sum()

        print(
            '\n{} unique customers in train set. Ratio: {}.'
            .format(uniq_visitors, (uniq_visitors / rows))
        )
        print(
            '{} instances have non-zero revenue out of {}. Ratio: {}.'
            .format(nzi, rows, (nzi / rows))
        )
        print(
            '{} unique customers with non-zero revenue out of {}. Ratio: {}.'
            .format(nzr, uniq_rows, (nzr / uniq_rows))
        )
        print('\nTotal columns with at least one Values:')
        print(df[~(df['Total'] == 0)])

        print('\nOriginal Dataframe Info:\n')

        print(train_df.info())
        print()

    def fill_na_values(self):
        train_df = self.train_df
        test_df = self.test_df

        na_val_keys = {
            'totals.pageviews': [1, 'int'],
            'totals.newVisits': [0, 'int'],
            'totals.bounces': [0, 'int'],
            'totals.transactionRevenue': [0, 'float'],
            'trafficSource.isTrueDirect': [False, 'bool'],
            'trafficSource.referralPath': ['N/A', 'str']
        }
        for nav_k, [nav_v, nav_t] in na_val_keys.items():
            train_df[nav_k] = train_df[nav_k].fillna(nav_v)
            train_df[nav_k] = train_df[nav_k].astype(nav_t)

            if nav_k != 'totals.transactionRevenue':
                test_df[nav_k] = test_df[nav_k].fillna(nav_v)
                test_df[nav_k] = test_df[nav_k].astype(nav_t)

        na_vals = [
            'unknown.unknown', '(not set)', 'not available in demo dataset',
            '(not provided)', '(none)', '<NA>', 'nan'
        ]

        for na_val in na_vals:
            train_df = train_df.replace(na_val, 'N/A')
            test_df = test_df.replace(na_val, 'N/A')

        self.train_df = train_df
        self.test_df = test_df

    def remove_cols(self):
        print('remove_cols')

        train_df = self.train_df
        test_df = self.test_df

        train_df_cols = train_df.columns
        test_df_cols = test_df.columns

        const_cols = [
            c
            for c in train_df_cols
            if train_df[c].nunique(dropna=False) == 1
        ]
        const_cols += [
            'sessionId', 'visitId', 'trafficSource.adContent',
            'trafficSource.adwordsClickInfo.slot',
            'trafficSource.adwordsClickInfo.page',
            'trafficSource.adwordsClickInfo.gclId',
            'trafficSource.adwordsClickInfo.adNetworkType',
            'trafficSource.adwordsClickInfo.isVideoAd'
        ]

        cols_not_in_test = set(train_df_cols).difference(set(test_df_cols))

        train_const_cols = const_cols + ['trafficSource.campaignCode']

        train_df.drop(train_const_cols, axis=1, inplace=True)
        test_df.drop(const_cols, axis=1, inplace=True)

        print('\nTotal Train Features dropped : {}'.format(train_const_cols))
        print('\nTrain features dropped: {}'.format(len(train_const_cols)))

        print('\nTotal Test Features dropped : {}'.format(const_cols))
        print('\nTest features dropped: {}'.format(len(const_cols)))

        print('\nTrain Shape after dropping: {}'.format(train_df.shape))
        print('\nTest Shape after dropping: {}\n'.format(test_df.shape))

        print(train_df.info())
        print()

        self.train_df = train_df
        self.test_df = test_df

    def normalize(self):
        train_df = self.train_df
        test_df = self.test_df

        str_cols = [
            'channelGrouping', 'device.browser', 'device.deviceCategory',
            'device.isMobile', 'device.operatingSystem', 'geoNetwork.city',
            'geoNetwork.continent', 'geoNetwork.country', 'geoNetwork.metro',
            'geoNetwork.networkDomain', 'geoNetwork.region',
            'geoNetwork.subContinent', 'trafficSource.campaign',
            'trafficSource.isTrueDirect', 'trafficSource.keyword',
            'trafficSource.medium', 'trafficSource.source',
            'trafficSource.referralPath'
        ]
        num_cols = [
            'totals.bounces', 'totals.hits', 'totals.newVisits',
            'totals.pageviews', 'visitNumber', 'visitStartTime'
        ]

        train_core_df = train_df.loc[
            :,
            ['fullVisitorId', 'totals.transactionRevenue', 'date']
        ]
        test_core_df = test_df.loc[:, ['fullVisitorId', 'date']]

        train_rest_df = train_df.loc[:, str_cols + num_cols]
        test_rest_df = test_df.loc[:, str_cols + num_cols]

        for str_col in str_cols:
            labelEncoder = LabelEncoder()

            train_col_list = list(train_rest_df[str_col].astype('str'))
            test_col_list = list(test_rest_df[str_col].astype('str'))

            labelEncoder.fit(train_col_list + test_col_list)

            train_rest_df[str_col] = labelEncoder.transform(train_col_list)
            test_rest_df[str_col] = labelEncoder.transform(test_col_list)

        for num_col in num_cols:
            train_rest_df[num_col] = train_rest_df[num_col].astype('float')
            test_rest_df[num_col] = test_rest_df[num_col].astype('float')

        minMaxScaler = MinMaxScaler()

        normalized_train_df = pd.DataFrame(
            minMaxScaler.fit_transform(train_rest_df.astype('float'))
        )
        normalized_train_df.columns = train_rest_df.columns
        normalized_train_df.index = train_rest_df.index

        minMaxScaler = MinMaxScaler()

        normalized_test_df = pd.DataFrame(
            minMaxScaler.fit_transform(test_rest_df.astype('float'))
        )
        normalized_test_df.columns = test_rest_df.columns
        normalized_test_df.index = test_rest_df.index

        cleaned_train_df = pd.concat(
            [train_core_df, normalized_train_df],
            axis=1
        )

        cleaned_test_df = pd.concat(
            [test_core_df, normalized_test_df],
            axis=1
        )

        self.cleaned_train_df = cleaned_train_df
        self.cleaned_test_df = cleaned_test_df
        self.str_cols = str_cols
        self.num_cols = num_cols

    def split_data(self):
        cleaned_train_df = self.cleaned_train_df
        cleaned_test_df = self.cleaned_test_df
        str_cols = self.str_cols
        num_cols = self.num_cols

        print('\nTransactions Minimum Date: {}'.format(
            cleaned_train_df['date'].min()
        ))
        print('Transactions Maximum Date: {}'.format(
            cleaned_train_df['date'].max()
        ))

        cleaned_train_df["date"] = cleaned_train_df["date"]\
            .astype('str')\
            .apply(lambda x: parse(x, yearfirst=True))

        dev_df = cleaned_train_df[
            cleaned_train_df['date'] < datetime(2017, 6, 1)
        ]
        val_df = cleaned_train_df[
            cleaned_train_df['date'] >= datetime(2017, 6, 1)
        ]

        feature_cols = ['fullVisitorId'] + str_cols + num_cols

        dev_X = dev_df[['totals.transactionRevenue'] + feature_cols]
        val_X = val_df[['totals.transactionRevenue'] + feature_cols]
        test_X = cleaned_test_df[feature_cols]

        print('\nNumber of instances in train: {} with {} columns.'.format(
            dev_X.shape[0], dev_X.shape[1]
        ))
        print('\nNumber of instances in val: {} with {} columns.'.format(
            val_X.shape[0], val_X.shape[1]
        ))
        print('\nNumber of instances in test: {} with {} columns.'.format(
            test_X.shape[0], test_X.shape[1]
        ))
        self.dev_X = dev_X
        self.val_X = val_X
        self.test_X = test_X

    def write_to_csv(self):
        if not path.exists(self.files_dir):
            makedirs(self.files_dir)

        self.dev_X.to_csv(self.train_file, index=False, compression='zip')
        self.val_X.to_csv(self.val_file, index=False, compression='zip')
        self.test_X.to_csv(self.test_file, index=False, compression='zip')
