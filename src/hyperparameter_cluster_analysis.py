# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914

"""
Run clustering with benchmarks.

Adapted from
https://www.kaggle.com/code/hellbuoy/online-retail-k-means-hierarchical-clustering/notebook
"""

import os
import pathlib
import time

import argparse
import logging
import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from utils.preprocessing import generate_features


def main(flags):
    """Run learning with benchmarks.

    Args:
        flags: Run flags.
    """
    if flags.logfile == "":
        logging.basicConfig(level=logging.DEBUG)
    else:
        path = pathlib.Path(flags.logfile)
        path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=flags.logfile, level=logging.DEBUG)
    logger = logging.getLogger()

    # if using intel, patch sklearn first
    if flags.intel:
        logger.debug("Loading intel libraries...")

        from sklearnex import patch_sklearn
        patch_sklearn()

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans, DBSCAN

    # Read in the datasets
    logger.debug("Reading in the data...")

    if not os.path.exists("../data/OnlineRetail.csv"):
        logger.error(
            "Data file ../data/OnlineRetail.csv not found")
        return

    # Preprocess dataset to create features
    preprocessing_start = time.time()
    rfm_df = generate_features(
        num_repeats=FLAGS.repeats,
        use_small_features=FLAGS.use_small_features
    )
    preprocessing_time = time.time() - preprocessing_start

    logger.info('=======> Preprocessing Time : %d secs', preprocessing_time)
    logger.info(
        'Number of data points : (%d, %d)', rfm_df.shape[0], rfm_df.shape[1]
    )

    if flags.algo == 'kmeans':

        # hyperparameter tuning for KMeans clustering
        logger.info("Generating clusters using K-Means...")
        range_n_clusters = [2, 3, 4, 5, 10, 15, 20, 25, 30]
        range_tol = [1e-3, 1e-4, 1e-5]

        kmeans_times = []
        for tol in range_tol:
            for num_clusters in range_n_clusters:
                np.random.seed(42)
                start = time.time()
                kmeans_pipe = Pipeline([
                    ('scaler', StandardScaler()),
                    ('kmeans', KMeans(n_clusters=num_clusters,
                                      tol=tol, random_state=0))
                ])
                predictions = kmeans_pipe.fit_predict(rfm_df)
                end = time.time()

                if flags.save_model_dir:
                    # save model and predictions for each model
                    fname = f"model_{num_clusters}_{tol}.pkl"
                    pname = f"pred_{num_clusters}_{tol}.txt"
                    os.makedirs(
                        os.path.join(flags.save_model_dir, "kmeans"),
                        exist_ok=True
                    )
                    joblib.dump(
                        kmeans_pipe,
                        os.path.join(flags.save_model_dir,
                                     "kmeans", fname)
                    )

                    np.savetxt(
                        os.path.join(flags.save_model_dir, "kmeans", pname),
                        predictions.astype(int),
                        fmt='%i'
                    )
                    logger.info(
                        "Model saved to %s/kmeans/%s",
                        flags.save_model_dir,
                        fname
                    )
                    logger.info(
                        "Predictions saved to %s/kmeans/%s",
                        flags.save_model_dir,
                        pname
                    )

                logger.info(
                    '===> Training Time (clusters = %d, tol = %f) : %.3f secs',
                    num_clusters, tol, end - start
                )
                kmeans_times.append(end - start)

        if flags.save_model_dir:
            # save preprocessed data
            rfm_df.to_csv(
                os.path.join(flags.save_model_dir, "kmeans", "data.csv"),
                index=False
            )
        hyperparameter_time = np.sum(kmeans_times)
        logger.info(
            'Total hyperparameter tuning time : %.3f secs', hyperparameter_time
        )
    elif flags.algo == 'dbscan':

        # hyperparameter tuning for DBSCAN clustering
        logger.info("Generating clusters using DBSCAN...")
        range_min_samples = [10, 50, 100]
        range_eps = [0.3, 0.5, 0.7]

        dbscan_times = []
        for min_samples in range_min_samples:
            for eps in range_eps:
                np.random.seed(42)
                start = time.time()
                dbscan_pipe = Pipeline([
                    ('scaler', StandardScaler()),
                    ('dbscan', DBSCAN(eps=eps, min_samples=min_samples))
                ])

                dbscan_pipe.fit(rfm_df)
                end = time.time()
                predictions = dbscan_pipe.named_steps['dbscan'].labels_

                # save model
                if flags.save_model_dir:
                    # save model and predictions for each model
                    fname = f"model_{min_samples}_{eps}.pkl"
                    pname = f"pred_{min_samples}_{eps}.txt"
                    os.makedirs(
                        os.path.join(flags.save_model_dir, "dbscan"),
                        exist_ok=True
                    )

                    joblib.dump(
                        dbscan_pipe,
                        os.path.join(flags.save_model_dir,
                                     "dbscan", fname)
                    )

                    np.savetxt(
                        os.path.join(flags.save_model_dir, "dbscan", pname),
                        predictions.astype(int),
                        fmt='%i'
                    )
                    logger.info(
                        "Model saved to %s/dbscan/%s",
                        flags.save_model_dir,
                        fname
                    )
                    logger.info(
                        "Predictions saved to %s/dbscan/%s",
                        flags.save_model_dir,
                        pname
                    )

                logger.info(
                    '===> Training Time (min_samples = %d, eps = %f) : %.3f secs',
                    min_samples, eps, end - start)
                dbscan_times.append(end - start)

        if flags.save_model_dir:
            # save preprocessed data
            rfm_df.to_csv(
                os.path.join(flags.save_model_dir, "kmeans", "data.csv"),
                index=False
            )
        hyperparameter_time = np.sum(dbscan_times)
        logger.info(
            'Total hyperparameter tuning time : %.3f secs', hyperparameter_time
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-l',
                        '--logfile',
                        type=str,
                        default="",
                        help="log file to output benchmarking results to")

    parser.add_argument('-i',
                        '--intel',
                        default=False,
                        action="store_true",
                        help="use intel technologies where available"
                        )

    parser.add_argument('--use_small_features',
                        default=False,
                        action="store_true",
                        help="use 3 features instead of 21"
                        )

    parser.add_argument('-r',
                        '--repeats',
                        default=1,
                        type=int,
                        help="number of times to clone the data"
                        )

    parser.add_argument('-a',
                        '--algo',
                        default='kmeans',
                        type=str,
                        choices=['kmeans', 'dbscan'],
                        help="clustering algorithm to use"
                        )

    parser.add_argument('--save_model_dir',
                        default=None,
                        type=str,
                        help="directory to save ALL models if desired"
                        )

    FLAGS = parser.parse_args()
    main(FLAGS)
