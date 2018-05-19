#!/usr/bin/env python
"""
coding=utf-8
Code Template
"""

import pickle
import logging


import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import log_loss, accuracy_score, classification_report

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


# EXTRACT

data_dir = '../AirBnB_data/'
random_state = 42


# Read all tables from .csv into DataFrames:
def load_data_from_csv():
    """
    Load all five .csv data files and return as five separate Pandas
    DataFrames.
    :return: agb, countries, sessions, train_users, test_users
    """
    agb = pd.read_csv(data_dir + 'age_gender_bkts.csv')
    countries = pd.read_csv(data_dir + 'countries.csv')
    sessions = pd.read_csv(data_dir + 'sessions.csv')
    train_users = pd.read_csv(data_dir + 'train_users_2.csv')
    test_users = pd.read_csv(data_dir + 'test_users.csv')

    return agb, countries, sessions, train_users, test_users


def dump_pickle(data, filename):
    with open(filename, 'wb') as picklefile:
        pickle.dump(data, picklefile)


def load_pickle(file_pkl):
    with open(file_pkl, 'rb') as picklefile:
        return pickle.load(picklefile)


def load_data_from_pkl():
    agb = load_pickle(data_dir + 'agb.pkl')
    countries = load_pickle(data_dir + 'countries.pkl')
    sessions = load_pickle(data_dir + 'sessions.pkl')
    train_users = load_pickle(data_dir + 'train_users.pkl')
    test_users = load_pickle(data_dir + 'test_users.pkl')

    return agb, countries, sessions, train_users, test_users


# TRANSFORM:

def get_clean_agb(agb):
    """
    Drop extreme age buckets and convert age buckets to Pandas cut-like
    strings of the form [lower, higher).
    :param agb:
    :return: agb
    """
    # Drop age buckets that seem unreasonable or specious:
    agb = agb[~agb.age_bucket.isin(
        ['100+', '0-4', '5-9', '10-14', '95-99', '90-94', '85-89'])]

    #
    def transform_age_cuts(age_bucket):
        splitted = age_bucket.split('-')
        return '[' + splitted[0] + ', ' + str(int(splitted[1]) + 1) + ')'

    agb.loc[:, 'age_bucket'] = agb['age_bucket'].map(transform_age_cuts)

    # Add 'probability_given_gender_age' column to age_gender_brackets:
    sum_agb = agb.groupby(['age_bucket', 'gender'], as_index=False)[
        'population_in_thousands'].sum()

    new_agb = pd.merge(agb, sum_agb, on=['age_bucket', 'gender'])

    new_agb['probability_given_gender_age'] = new_agb[
                                                  'population_in_thousands_x'] / \
                                              new_agb[
                                                  'population_in_thousands_y']

    new_agb.drop('year', axis=1, inplace=True)
    new_agb.drop('population_in_thousands_y', axis=1, inplace=True)

    return new_agb


def get_clean_train_users(train_users):
    """
    Accept raw train_users and return df without NaNs, and with parsable time data, and with reasonable age limits.
    """
    if 'date_account_created' in train_users.columns:
        date_account = np.vstack(train_users['date_account_created'].astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
        train_users['date_account_created_year'] = date_account[:, 0]
        train_users['date_account_created_month'] = date_account[:, 1]
        # train_users['date_account_created_day'] = date_account[:, 2] # drop
        train_users = train_users.drop(['date_account_created'], axis=1)

    # Process timestamp_first_active:
    if 'timestamp_first_active' in train_users.columns:
        tfa = np.vstack(train_users['timestamp_first_active'].astype(str)
                        .apply(lambda x: list(map(int, [x[:4], x[4:6], x[6:8],
                                                        x[8:10], x[10:12],
                                                        x[12:14]]))).values
                        )

        train_users['first_active_year'] = tfa[:, 0]
        train_users['first_active_month'] = tfa[:, 1]
        # train_users['first_active_day'] = tfa[:, 2] # drop first_active_day
        train_users = train_users.drop(['timestamp_first_active'], axis=1)

    # Drop date_first_booking (for NDF, date_first_booking is NaN):
    if 'date_first_booking' in train_users.columns:
        train_users.drop('date_first_booking', axis=1, inplace=True)

    # Constrain train users to reasonable ages, drop all else.
    train_users = train_users[(train_users['age'] <= 84)
                              & (train_users['age'] >= 15)
                              ]
    train_users['age_bucket'] = pd.cut(train_users.age,
                                        bins=list(range(15, 90, 5)),
                                        right=False,
                                        retbins=False).astype(str)

    # Combine -unknown- gender with OTHER into OTHER:
    train_users['gender'].replace('-unknown-', 'OTHER', inplace=True)

    train_users.loc[:, 'gender'] = train_users['gender'].map(
        lambda s: s.lower())

    # Drop country_destination NDF (no travel was booked):
    train_users = train_users[~(train_users['country_destination'] == 'NDF')]

    # Drop low-frequency rows:
    threshold = 65  # Anything that occurs less than this will be removed.
    for col in ['language', 'affiliate_provider', 'first_affiliate_tracked',
                'first_browser']:
        value_counts = train_users[col].value_counts()  # Specific column
        to_remove = value_counts[value_counts <= threshold].index
        for rem in to_remove:
            train_users[col].replace(rem, np.nan, inplace=True)

    # Drop all remaining rows that have NaNs:
    train_users.dropna(axis=0, inplace=True)

    return train_users


def join_sessions_aggs_on_train_users(train_users, sessions):
    """
    Aggregate select features from sessions dataframe and merge with train_users,
    creating an expanded train_users df.

    :param train_users: df.
    :param sessions: df.
    :return: joined_df, which is train_users with additional column(s).
    """

    message_requests = (sessions[sessions['action_type'] == 'message_post']
                        .groupby('user_id')['action_type'].count()
                        )

    joined_df = train_users.join(message_requests, on=['id'], how='left')

    joined_df.rename({'action_type': 'message_requests'}, axis='columns',
                     inplace=True)

    joined_df['message_requests'].fillna(0, inplace=True)

    joined_df.drop('id', axis=1, inplace=True)

    return joined_df


def join_agb_on_train_users(train_users, agb):

    pivoted = agb.pivot_table(columns='country_destination',
                              values='probability_given_gender_age',
                              index=['age_bucket', 'gender'])

    pivoted = pivoted.reset_index()

    joined_df = train_users.merge(pivoted, on=['age_bucket', 'gender'])

    joined_df.drop('age_bucket', axis=1, inplace=True)

    return joined_df


def get_train_test_split(joined_df):

    y = joined_df['country_destination']
    X = joined_df.drop('country_destination', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=random_state)

    return X_train, X_test, y_train, y_test

# FEATURE SELECTION:


def get_df_with_dummies(df):
    """
    Converts df to df_with_dummies based on column dtype. Column with dtype
    "object" will be One-hot-encoded, all else left the same.
    :param df:
    :return: df_with_dummies
    """
    df_with_dummies = pd.get_dummies(df,
                            columns=df.select_dtypes(include='object').columns,
                            prefix=df.select_dtypes(include='object').columns,
                                     )

    return df_with_dummies


def binarize_y(y_raw, binary_country=None):
    """
    Takes raw, multi-class pandas series and returns binarized series based on
    "country or not" rule.
    :param y_raw: DataFrame.
        I.e.: test_users['country_destination']
    :param binary_country: str.
        E.g. 'US', 'DE', 'NDF'.
    :return: pandas series.
        Same shape and format as y_raw, but with 1s and 0s.
    """

    if binary_country is not None:
        y_binary = (y_raw == binary_country).astype(int)
    else:
        y_binary = y_raw
    return y_binary


def encode_X(X_raw, drop_key='drop_list_1'):
    """
    Accept X_raw df and drop_key and return X df.

    :param X_raw: DataFrame.
    :param drop_key: str.
    :return: X, DataFrame.
    """
    drop_dict = {'drop_list_1': ['affiliate_channel',
                                 'first_affiliate_tracked',
                                 'signup_app', 'first_browser',
                                 'affiliate_provider', 'first_device_type',
                                 ]
                 }

    # Down-select cols first:
    if 'id' in X_raw.columns:
        X_raw.drop('id', axis=1, inplace=True)

    for col in drop_dict[drop_key]:
        if col in X_raw.columns:
            X_raw.drop(col, axis=1, inplace=True)

    X = get_df_with_dummies(X_raw)

    return X


# LOAD:


def train_classifiers(X_train, X_test, y_train, y_test):
    """
    DEPRECATED.
    """
    clfs = {'LR': LogisticRegression(random_state=random_state),
            'SVM': SVC(probability=True, random_state=random_state),
            'RF': RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                         random_state=random_state),
            # 'GBM': GradientBoostingClassifier(n_estimators=50,
            #                                  random_state=random_state),
            # 'ETC': ExtraTreesClassifier(n_estimators=100, n_jobs=-1,
            #                            random_state=random_state),
            'KNN': KNeighborsClassifier(n_neighbors=30)}

    # predictions on the validation and test sets
    y_test_list = []
    scores = []

    print('Performance of individual classifiers on X_test')
    print('------------------------------------------------------------')

    for name, clf in clfs.items():
        # First run. Training on (X_train, y_train) and predicting on X_valid.
        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_test)
        y_test_list.append(y_pred)

        # Printing out the performance of the classifier
        print('{:10s} {:2s} {:1.7f}'.format('%s: ' % (name), 'logloss  =>',
                                            log_loss(y_test, y_pred)))

        print('{:10s} {:2s} {:1.2f}'.format('%s: ' % (name), 'accuracy  =>',
                                            accuracy_score(y_test, y_pred)))

    print('')

    return None


def main(from_pickle=True):

    logging.basicConfig(level=logging.DEBUG)

    logging.info(' Begin extract')
    if not from_pickle:
        agb, countries, sessions, train_users, test_users = load_data_from_csv()
    else:
        agb, countries, sessions, train_users, test_users = load_data_from_pkl()

    # Clean train_users and age_gender_bracket tables:
    logging.info(' Begin transform')
    train_users = get_clean_train_users(train_users)
    agb = get_clean_agb(agb)

    # Merge sessions and age_gender_brackets data with train_users
    train_users = join_sessions_aggs_on_train_users(train_users, sessions)
    train_users = join_agb_on_train_users(train_users, agb)

    # Do Train Test split:
    logging.info(' Train Validation Holdout Split')
    holdout_set = train_users.sample(frac=0.2, random_state=random_state)
    remainder_set = train_users.drop(holdout_set.index)
    validation_set = remainder_set.sample(frac=0.25, random_state=random_state)
    training_set = train_users.drop(validation_set.index)

    logging.info(' Return: ')
    return training_set, validation_set, holdout_set, agb, sessions


# Main section
if __name__ == '__main__':
    main()
