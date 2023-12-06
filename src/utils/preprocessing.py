# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914

"""
Methods to do data pre-processing and feature engineering.
"""

import pandas as pd
import numpy as np


def generate_features(dataset_path:str, num_repeats: int = 1, use_small_features=False) -> pd.DataFrame:
    """Read data and extract features.  Repeats the dataset `num_repeats` times.

    Args:
        dataset_path (str):
            Path to where the dataset is located.
        num_repeats (int, optional):
            The number of times to repeat the dataset. Defaults to 1.
        use_small_features (bool, optional):
            Use 3 features instead of 21.

    Returns:
        pd.DataFrame: Dataframe with extracted features.
    """
    retail = pd.read_csv(dataset_path,
                         encoding="ISO-8859-1")

    # Create relevant data frames
    retail = retail.dropna()
    retail['CustomerID'] = retail['CustomerID'].astype(str)
    retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'])
    retail['Month'] = retail['InvoiceDate'].dt.month_name()
    retail['DayOfWeek'] = retail['InvoiceDate'].dt.day_name()
    retail['Diff'] = max(retail['InvoiceDate']) - retail['InvoiceDate']
    retail['Amount'] = retail['Quantity'] * retail['UnitPrice']

    # average spending of each customer
    rfm_m = retail.groupby('CustomerID')['Amount'].mean()
    rfm_m.reset_index()

    # total number of purchases per customer
    rfm_f = retail.groupby('CustomerID')['InvoiceNo'].count()
    rfm_f = rfm_f.reset_index()
    rfm_f.columns = ['CustomerID', 'NumberOfOrders']

    # purchases per day of week
    rfm_dw = retail.groupby('CustomerID')[
        'DayOfWeek'].value_counts().unstack(-1).fillna(0)

    # purchases per month of year
    rfm_my = retail.groupby('CustomerID')[
        'Month'].value_counts().unstack(-1).fillna(0)

    # recency of purchases
    rfm_p = retail.groupby('CustomerID')['Diff'].min()
    rfm_p = rfm_p.reset_index()
    rfm_p['Recency'] = rfm_p['Diff'].dt.days

    # Collect all of the datasets together
    rf_combined = pd.merge(rfm_m, rfm_f, on='CustomerID', how='inner')
    rf_combined = pd.merge(rf_combined, rfm_p, on='CustomerID', how='inner')
    rf_combined = pd.merge(rf_combined, rfm_dw, on='CustomerID', how='inner')
    rf_combined = pd.merge(rf_combined, rfm_my, on='CustomerID', how='inner')

    # Removing (statistical) outliers for Amount
    q_1 = rf_combined.Amount.quantile(0.05)
    q_3 = rf_combined.Amount.quantile(0.95)
    iqr = q_3 - q_1
    rf_combined = rf_combined[
        (rf_combined.Amount >= q_1 - 1.5*iqr) &
        (rf_combined.Amount <= q_3 + 1.5*iqr)
    ]

    # Removing (statistical) outliers for Recency
    q_1 = rf_combined.Recency.quantile(0.05)
    q_3 = rf_combined.Recency.quantile(0.95)
    iqr = q_3 - q_1
    rf_combined = rf_combined[
        (rf_combined.Recency >= q_1 - 1.5*iqr) &
        (rf_combined.Recency <= q_3 + 1.5*iqr)
    ]

    # Removing (statistical) outliers for Frequency
    q_1 = rf_combined.NumberOfOrders.quantile(0.05)
    q_3 = rf_combined.NumberOfOrders.quantile(0.95)
    iqr = q_3 - q_1
    rf_combined = rf_combined[
        (rf_combined.NumberOfOrders >= q_1 - 1.5*iqr) &
        (rf_combined.NumberOfOrders <= q_3 + 1.5*iqr)
    ]
    rf_combined = rf_combined.loc[rf_combined.index.repeat(num_repeats)]

    # use small or large feature sets
    if use_small_features:
        columns = ['Amount', 'NumberOfOrders', 'Recency']
    else:
        columns = ['Amount', 'NumberOfOrders', 'Recency', 'Friday',
                   'Monday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday',
                   'April', 'August', 'December', 'February', 'January',
                   'July', 'June', 'March', 'May', 'November', 'October',
                   'September']

    rfm_df = rf_combined[columns]

    # add jitter to the entries so we avoid identical entries since we
    # duplicate data
    if num_repeats > 1:
        new_amount = rfm_df['Amount'] + \
            np.exp(np.random.normal(0, 1, size=rfm_df.shape[0]))
        cat_cols = rfm_df[[col for col in columns if col != 'Amount']]
        new_cat = np.random.choice([0, 1, 2, 3, 4],
                                   size=cat_cols.shape) + cat_cols
        rfm_df = pd.concat([new_amount, new_cat], axis=1)

    rfm_df.columns = columns

    return rfm_df
