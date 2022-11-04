#!/bin/bash

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

for j in 1 10 100
do
	for n in {1..3}
	do
		echo "Starting r = $j , n = $n, algo = kmeans"
		python hyperparameter_cluster_analysis.py -r $j -l ../logs/kmeans/intel_large_$j.log -i -a kmeans
	done
done

for j in 1 10 15
do
	for n in {1..3}
	do
		echo "Starting r = $j , n = $n, algo = dbscan"
		python hyperparameter_cluster_analysis.py -r $j -l ../logs/dbscan/intel_large_$j.log -i -a dbscan
	done
done

