
import pandas as pd
import json
# import numpy as np
# import matplotlib.pyplot as plt

from dateutil.parser import parse
from datetime import datetime
from os import getcwd
from os.path import join, basename
from pandas.io.json import json_normalize
from sklearn.preprocessing import LabelEncoder


class ProcessData:
    def __init__(self, nrows=3000):
        self.nrows = nrows
        self.input_train_file = join(getcwd(), 'data/orig/transactions.csv')

        self.train_file = join(getcwd(), 'data/files/train.csv')
        self.val_file = join(getcwd(), 'data/files/val.csv')
        self.test_file = join(getcwd(), 'data/files/test.csv')

    def write_to_csv(self, df, csv_path):
        df.to_csv(csv_path, index=False)

    def load(self, nrows=1000):
        print('load')
        JSON_COLUMNS = [
            'device', 'geoNetwork', 'totals', 'trafficSource'
        ]

        df = pd.read_csv(
            self.input_train_file,
            converters={column: json.loads for column in JSON_COLUMNS},
            dtype={'fullVisitorId': 'str'},
            nrows=nrows
        )

        print(df.columns)

        for column in JSON_COLUMNS:
            column_as_df = json_normalize(df[column])
            column_as_df.columns = [
                f'{column}.{subcolumn}'
                for subcolumn in column_as_df.columns
            ]
            df = df.drop(column, axis=1)
            df = df.merge(column_as_df, right_index=True, left_index=True)

        [rows, columns] = df.shape
        print(df.columns)

        print(df['trafficSource.referralPath'].unique())

        print(
            '\nLoaded {} rows with {} columns from {}.\n'.format(
                rows, columns, basename(self.input_train_file)
            )
        )

        self.train_df = df

    def missing_data(self):
        print('missing_data')
        train_df = self.train_df
        visitor_id = 'fullVisitorId'
        revenue_id = 'totals.transactionRevenue'

        total_null_sum = train_df.isnull().sum()
        total_null_count = train_df.isnull().count()

        dfgv_id = train_df.groupby(visitor_id)
        dfgv_id = dfgv_id[revenue_id].sum().reset_index()

        total = total_null_sum.sort_values(ascending=False)
        percent = (total_null_sum / total_null_count * 100)
        percent = percent.sort_values(ascending=False)

        df = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        rows = train_df.shape[0]
        uniq_rows = dfgv_id.shape[0]

        uniq_visitors = train_df[visitor_id].nunique()

        nzi = len(
            train_df[revenue_id].to_numpy().nonzero()[0]
        )
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

        # print(train_df.info())

    def fill_na_values(self):
        print('fill_na_values')
        train_df = self.train_df

        na_val_keys = {
            'totals.pageviews': [1, 'int'],
            'totals.newVisits': [0, 'int'],
            'totals.bounces': [0, 'int'],
            'totals.transactionRevenue': [0, 'float'],
            'trafficSource.isTrueDirect': [False, 'bool'],
            'trafficSource.adwordsClickInfo.isVideoAd': [True, 'bool']
        }
        for nav_k, [nav_v, nav_t] in na_val_keys.items():
            train_df[nav_k] = train_df[nav_k].fillna(nav_v)
            train_df[nav_k] = train_df[nav_k].astype(nav_t)

        na_vals = [
            'unknown.unknown', '(not set)', 'not available in demo dataset',
            '(not provided)', '(none)', '<NA>', 'nan'
        ]

        for na_val in na_vals:
            train_df = train_df.replace(na_val, 'N/A')

        self.train_df = train_df

    def remove_cols(self):
        print('remove_cols')

        train_df = self.train_df

        const_cols = [
            c
            for c in train_df.columns
            if train_df[c].nunique(dropna=False) == 1
        ]
        const_cols += [
            'hits', 'customDimensions', 'trafficSource.adContent',
            'trafficSource.adwordsClickInfo.slot',
            'trafficSource.adwordsClickInfo.page',
            'trafficSource.adwordsClickInfo.gclId',
            'trafficSource.adwordsClickInfo.adNetworkType',
            'trafficSource.referralPath', 'trafficSource.campaign',
            'trafficSource.keyword'
        ]

        train_df.drop(const_cols, axis=1, inplace=True)

        self.train_df = train_df

    def normalize(self):
        print('normalizing')
        train_df = self.train_df

        str_cols = [
            'channelGrouping', 'device.browser', 'device.deviceCategory',
            'device.isMobile', 'device.operatingSystem', 'geoNetwork.city',
            'geoNetwork.continent', 'geoNetwork.country',
            'geoNetwork.metro', 'geoNetwork.networkDomain',
            'geoNetwork.region', 'geoNetwork.subContinent',
            'totals.timeOnSite', 'trafficSource.adwordsClickInfo.isVideoAd',
            'trafficSource.isTrueDirect', 'trafficSource.medium',
            'trafficSource.source'
        ]
        num_cols = [
            'totals.bounces', 'totals.hits', 'totals.newVisits',
            'totals.pageviews', 'totals.sessionQualityDim',
            'totals.transactions', 'totals.timeOnSite'
        ]

        for str_col in str_cols:
            labelEncoder = LabelEncoder()
            col_str_list = train_df[str_col].astype('str')

            labelEncoder.fit(col_str_list)

            train_df[str_col] = labelEncoder.transform(col_str_list)

        for num_col in num_cols:
            train_df[num_col] = train_df[num_col].astype('float')

        # minMaxScaler = MinMaxScaler()

        # normalized_df = pd.DataFrame(
        #     minMaxScaler.fit_transform(train_df.astype('float'))
        # )

        # normalized_df.columns = train_df.columns
        # normalized_df.index = train_df.index

        # train_df_cols = list(train_df.columns)
        # for remove_col in ['totals.transactionRevenue', 'fullVisitorId']:
        #     train_df_cols.remove(remove_col)

        # train_df_cols.sort()

        # map_train_df_cols = {
        #     'fullVisitorId': 'f_1',
        #     'totals.transactionRevenue': 'f_{}'.format(train_df.shape[1])
        # }

        # for idx, train_df_col in enumerate(train_df_cols, start=1):
        #     map_train_df_cols[train_df_col] = 'f_{}'.format(idx + 1)

        # normalized_df = normalized_df.rename(columns=map_train_df_cols)
        # train_df['date'] = pd.to_datetime(train_df['date'])

        train_df["date"] = train_df["date"].astype('str').apply(
            lambda x: parse(x, yearfirst=True)
        )

        train_df_date = train_df['date']

        train_data = train_df[train_df_date < datetime(2017, 11, 1)]
        val_data = train_df[
            train_df_date > datetime(2017, 10, 31) and
            train_df_date < datetime(2017, 3, 1)
        ]
        test_data = train_df[train_df_date >= datetime(2017, 3, 1)]

        feature_cols = str_cols + num_cols

        train_data = train_data[feature_cols + ['totals.transactionRevenue']]
        val_data = val_data[feature_cols + ['totals.transactionRevenue']]
        test_data = test_data[feature_cols]

        self.write_to_csv(train_data, self.train_file)
        self.write_to_csv(val_data, self.val_file)
        self.write_to_csv(test_data, self.test_file)

        print('train data shape: ', train_data.shape)
        print('val data shape: ', val_data.shape)
        print('test data shape: ', test_data.shape)
        print('train_df shape: ', train_df.shape)

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.train_df = train_df

urce.adContent', 'trafficSource.adwordsClickInfo.slot',
    'trafficSource.adwordsClickInfo.page', 'trafficSource.adwordsClickInfo.gclId',
    'trafficSource.adwordsClickInfo.adNetworkType', 'trafficSource.adwordsClickInfo.isVideoAd',
    'trafficSource.campaignCode', 'trafficSource.keyword'
]
