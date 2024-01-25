#!/bin/bash

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

for j in 1 10 100
do
	for n in {1..3}
	do
		echo "Starting r = $j , n = $n, algo = kmeans"
		python $WORKSPACE/src/hyperparameter_cluster_analysis.py -r $j -l $OUTPUT_DIR/logs/kmeans/intel_large_$j.log -a kmeans --dataset_path $DATA_DIR/OnlineRetail.csv
	done
done

for j in 1 10 15
do
	for n in {1..3}
	do
		echo "Starting r = $j , n = $n, algo = dbscan"
		python $WORKSPACE/src/hyperparameter_cluster_analysis.py -r $j -l $OUTPUT_DIR/logs/dbscan/intel_large_$j.log -a dbscan --dataset_path $DATA_DIR/OnlineRetail.csv
	done
done

