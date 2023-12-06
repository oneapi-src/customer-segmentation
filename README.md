# **Customer Segmentation**

## Introduction

This reference kit provides a Machine Learning (ML) workflow to segment customers into clusters using Intel® Extension for Scikit-learn*. Customer segmentation helps to build a deeper understanding of a businesses clientele and can be further used to implement personalized targeted campaigns.

Check out more workflow examples in the [Developer Catalog](https://developer.intel.com/aireferenceimplementations).

## Solution Technical Overview

The growing need for customer analytics in enhancing customer experience and loyalty through predictive analysis, as well as personalizing marketing on the past data of the potential customers, is driving market growth. Customers expect to be considered as individuals with unique interests, which has shifted the emphasis to personalized brand experiences. This leads to the need to create a unified view of the customer as they connect with a brand and personalize the experience of consumers through networks, locations, and always at the moment. Artificial Intelligence (AI) and ML techniques offer a promising opportunity to help analyze, and understand the different types and patterns of customers within an ecosystem.

The major factors driving the growth of the customer analytics market size include the need to understand customer buying behavior for a more personalized experience and the advent of resource-intensive technologies, such as AI, ML, and business process automation, to streamline marketing operations.

Customer analytics will evolve from retrospective analysis to real-time, behavior-driven interaction to achieve a personalized customer experience. Clients using these solutions must re-train their models to deal with ever-increasing and transforming data sets, as well as exploring their current datasets under different lens, if they want their investment to keep producing value.

One of the primary methods for deriving an understanding of customer segments is by analyzing and exploring different AI based clustering algorithms on various feature sets to identify key candidates and customer phenotypes:

> Customer Dataset => Repeated Cluster Analysis => Targeted Campaigns and Opportunities

The AI opportunity presented here is a system for performing and generating the many cluster solutions for a dataset, which can then be further explored by an analyst to provide insights.

Although AI delivers a solution to address target recommendation, on a production scale implementation with millions or billions of records demands for more compute power without leaving any performance on the table. Under this scenario, a business analyst aims to do a deep exploratory dive into their customer purchase data to find better opportunities. In order to derive the most insightful and beneficial actions to take, they will need to study and analyze the clusters generated through various feature sets and algorithms, thus requiring frequent re-runs of the algorithms under many different parameter sets. To utilize all the hardware resources efficiently, software optimizations cannot be ignored.

This workflow implementation is a reference solution to the described use case that includes an Optimized reference end-to-end (E2E) architecture enabled with Intel® Scikit-learn* Extension available as part of Intel® oneAPI AI toolkit optimizations.

Intel® Distribution for Python*, Intel® Extension for Scikit-learn* and Intel® Distribution of Modin* are used to optimize this pipeline for faster performance:

* ***Intel® Distribution for Python****

  The [Intel® Distribution for Python*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-python.html#gs.52te4z) provides:

  * Scalable performance using all available CPU cores on laptops, desktops, and powerful servers
  * Support for the latest CPU instructions
  * Near-native performance through acceleration of core numerical and machine learning packages with libraries like the Intel® oneAPI Math Kernel Library (oneMKL) and Intel® oneAPI Data Analytics Library
  * Productivity tools for compiling Python code into optimized instructions
  * Essential Python bindings for easing integration of Intel® native tools with your Python* project

* ***Intel® Extension for Scikit-learn****

  [Intel® Extension for Scikit-learn*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/scikit-learn.html) is a seamless way to speed up your Scikit-learn* applications for machine learning to solve real-world problems. This extension package dynamically patches Scikit-learn* estimators to use Intel® oneAPI Data Analytics Library (oneDAL) as the underlying solver, while achieving the speed up for your machine learning algorithms.

* ***Intel® Distribution of Modin****

  Modin* is a drop-in replacement for pandas, enabling data scientists to scale to distributed DataFrame processing without having to change API code. [Intel® Distribution of Modin*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-of-modin.html) adds optimizations to further accelerate processing on Intel hardware.

## Solution Technical Details

### Dataset

The dataset used for this reference kit is a set of 500k transactions covering 4000 customers from a United Kingdom multinational online retailer.  The dataset was collected over the span of 1 year and can be found at https://archive.ics.uci.edu/dataset/352/online+retail.  Instructions for downloading the data for use can be found at the `data/README.md` file or in [Download the Datasets](#download-the-datasets) section.

Before clustering analysis, the data is preprocessed to focus on customer purchasing behavior and a feature store of 21 features are generated during this process, including features such as purchase frequency and preference for day of the week.

> **Please see this data set's applicable license for terms and conditions. Intel Corporation does not own the rights to this data set and does not confer any rights to it.**

## Validated Hardware Details

[Intel® oneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html) is used to achieve quick results even when the data for a model are huge. It provides the capability to reuse the code present in different languages so that the hardware utilization is optimized to provide these results.

| Recommended Hardware
| ----------------------------
| CPU: Intel® 2th Gen Xeon® Platinum 8280 CPU @ 2.70GHz or higher
| RAM: 187 GB
| Recommended Free Disk Space: 20 GB or more

Code was tested on Ubuntu\* 22.04 LTS.

## How it Works

The following diagram shows the customer segmentation solution E2E workflow:

![Use_case_flow](assets/e2e-flow-optimized.png)

The following are some examples of expected inputs and outputs:

<div align="center">

**Input**                                 | **Output** |
| :---: | :---: |
| Customer Features | Cluster Assignments for Each Customer |

**Example Input**                                 | **Example Output** |
| :---: | :---: |
| ***ID***, ***Attribute 1***, ***Attribute 2*** <br> 1, 100, 7.2 <br> 2, 10, 1.3 <br> 3, 75, 4.5 <br> 4, 25, 0.2 |***ID***, ***Cluster Assignment*** <br> 1, 1 <br> 2, 2 <br> 3, 1 <br> 4, 2 |
</div>
From the above example it can be seen that the model's output matches the ID from the customer's features to a specific cluster number.

### Hyperparameter Cluster Analysis

Rather than providing a single clustering solution, in realistic scenarios, an analyst will need to run the same clustering algorithm multiple times on the same dataset, scanning across different hyperparameters to find the most meaningful set of clusters.  To capture this, our reference solution scans across a grid of hyperparameters for the selected algorithm, and generates a clustering solution at each of these points, which is defined as hyperparameter cluster analysis.  At each hyperparameter setting, the clustering solution and the trained model is saved for analysis.  In practice, the results at each hyperparameter setting provides the analyst with different segmentations of the data that they can take and further analyze.

## Get Started

Start by defining an environment variable that will store the workspace path, these directories will be created in further steps and will be used for all the commands executed using absolute paths.

[//]: # (capture: baremetal)
```bash
export WORKSPACE=$PWD/customer-segmentation
export DATA_DIR=$WORKSPACE/data
export OUTPUT_DIR=$WORKSPACE/output
```

## Download the Workflow Repository

Create a working directory for the workflow and clone the [main repository](https://github.com/intel-innersource/frameworks.ai.platform.sample-apps.customer-segmentation) into your working directory.

[//]: # (capture: baremetal)
```bash
mkdir -p $WORKSPACE && cd $WORKSPACE
```

```bash
git clone https://github.com/intel-innersource/frameworks.ai.platform.sample-apps.customer-segmentation.git .
```

[//]: # (capture: baremetal)
```bash
mkdir -p $DATA_DIR $OUTPUT_DIR/logs $OUTPUT_DIR/models
```

### Set Up Conda

To learn more, please visit [install anaconda on Linux](https://docs.anaconda.com/free/anaconda/install/linux/).
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

### Set Up Environment

The conda yaml dependencies are kept in `$WORKSPACE/env/intel_env.yml`.

| **Packages required in YAML file:**                 | **Version:**
| :---                          | :--
| `python`  | 3.10
| `intelpython3_full`  | 2024.0.0
| `modin-all`  | 0.24.1
| `xlsx2csv` | 0.8.1

Follow the next steps for Intel® Python* Distribution setup inside conda environment:

```bash
# If you have conda 23.10.0 or greater you can skip the following two lines
# since libmamba is already set as the default solver.
conda install -n base conda-libmamba-solver -y
conda config --set solver libmamba
conda env create -f $WORKSPACE/env/intel_env.yml
```

Environment setup is required only once.  Make sure no conda environment exists with the same name since this step does not cleanup/overwrite the existing environment. During this setup a new conda environment will be created with the dependencies listed in the YAML file.

Once the appropriate environment is created it has to be activated using the conda command as given below:

```bash
conda activate customer_segmentation_intel
```

### Download the Datasets

Download and preprocess the dataset:

[//]: # (capture: baremetal)
```bash
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx -P $DATA_DIR
xlsx2csv $DATA_DIR/Online\ Retail.xlsx $DATA_DIR/OnlineRetail.csv
```

## Supported Runtime Environment

You can execute the references pipelines using the following environments:

* Bare Metal

---

### Run Using Bare Metal

#### Set Up System Software

Our examples use the `conda` package and environment on your local computer. If you don't already have `conda` installed or the `conda` environment created, go to [Set Up Conda*](#set-up-conda) or see the [Conda* Linux installation instructions](https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html).

### Run Workflow

This customer segmentation approach uses [k-means](https://scikit-learn.org/stable/modules/clustering.html#k-means) and Density-Based Spatial Clustering of Applications with Noise ([DBSCAN](https://scikit-learn.org/stable/modules/clustering.html#dbscan)) from Scikit-learn* library to train an AI model and generate cluster labels for the passed in data.  This process is captured within the `hyperparameter_cluster_analysis.py` script. This script reads and preprocesses the data, and performs hyperparameter cluster analysis on either k-means or DBSCAN, while also reporting on the execution time for preprocessing and hyperparameter cluster analysis steps. Furthermore, this script can also save each of the intermediate models/cluster labels for an in-depth analysis of the quality of fit.  

The script takes the following arguments:

```bash
usage: hyperparameter_cluster_analysis.py [-h] [-l LOGFILE] [-i] [--use_small_features] [-r REPEATS] [-a {kmeans,dbscan}] [--save_model_dir SAVE_MODEL_DIR]

optional arguments:
  -h, --help            
                        show this help message and exit
  -l, --logfile LOGFILE
                        log file to output benchmarking results to (default: None)
  --use_small_features  
                        use 3 features instead of 21 (default: False)
  -r, --repeats REPEATS
                        number of times to clone the dataset (default: 1)
  -a, --algo {kmeans,dbscan}
                        clustering algorithm to use (default: kmeans)
  --save_model_dir SAVE_MODEL_DIR
                        directory to save ALL models if desired (default: None)
  --dataset_path DATASET_PATH   
                        directory where the dataset is located (default: ../data/OnlineRetail.csv)
```

As an example of using this, we can run the following commands:

[//]: # (capture: baremetal)
```bash
python $WORKSPACE/src/hyperparameter_cluster_analysis.py --logfile $OUTPUT_DIR/logs/output.log --algo kmeans --save_model_dir $OUTPUT_DIR/models --dataset_path $DATA_DIR/OnlineRetail.csv
```

This will perform hyperparameter cluster analysis using k-means/DBSCAN for the provided data, saving the data to the `$OUTPUT_DIR/models` directory and providing performance logs on the algorithm to the `$OUTPUT_DIR/logs/output.log` file.  More generally, this script serves to create k-means/DBSCAN models on 21 features, scanning across various hyperparameters ([such as cluster size for k-means and min_samples for DBSCAN](#hyperparameters)) for each of the models and saving a model at EACH hyperparameter setting to the provided `--save_model_dir` directory.

In a realistic pipeline, this process would follow the diagram described in [How it Works](#how-it-works), adding either a human in the loop to determine the quality of the clustering solution at each hyperparameter setting, or by adding heuristic measure to quantify cluster quality.  In this situation, we do not implement a clustering quality and instead save the trained models/predictions in the `--save_model_dir` directory at each hyperparameter setting for future analysis and cluster comparisons.

As an example of a possible clustering metric, Silhouette analysis is often used for k-means to help select the number of clusters.  See [here](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html) for further implementation details.  For example, this can also be used in the above script by adding a rough heuristic that only saves models above a certain heuristic score.

#### Running Cluster Analysis/Predictions

The above script will train and save models at different hyperparameter configurations for k-means or DBSCAN.  In addition to saving the models to `$OUTPUT_DIR/models`, the script will also save the following files for each hyper parameter configuration:

1. `$OUTPUT_DIR/models/{algo}/data.csv` - preprocessed data file 
2. `$OUTPUT_DIR/models/{algo}/model_{hyperparameters}.pkl` - trained model file 
3. `$OUTPUT_DIR/{algo}/pred_{hyperparameters}.txt` - cluster labels for each datapoint in the data file

These files can be used to analyze each of the clustering solutions generated from the hyperparameter cluster analysis.

For demonstrational purposes of Intel® Extension for Scikit-learn*, we benchmark a **hyperparameter cluster analysis** under the following data-augmentation transformations:

1. Using 21 features
2. Replicating and jittering the data with noise to have up to 400k rows (depending on algorithm)
   1. k-means - 40k, 400k samples
   2. DBSCAN - 40k, 60k samples

<a id='hyperparameters'></a>
The hyperparameters for each algorithm include:

##### k-means

**n_clusters**  | **tol** |                             
| :---: | :---: |
2, 3, 4, 5, 10, 15, 20, 25, 30 | 1e-3, 1e-4, 1e-5 |

##### DBSCAN

**min_samples**  | **eps** |                              
| :---: | :---: |
10, 50, 100             | 0.3, 0.5, 0.7 |

Noise is added to ensure that no two rows are exactly the same after replication. DBSCAN testing is limited to 60k samples for because of memory constraints on the machines used for testing.

To run the demonstration experiment, execute the following command:

[//]: # (capture: baremetal)
```bash
bash $WORKSPACE/src/run_exp.sh
```

### Clean Up Bare Metal

Before proceeding to the cleaning process, it is strongly recommended to make a backup of the data that the user wants to keep. To clean the previously downloaded and generated data, run the following commands:

```bash
conda activate base
conda env remove -n customer_segmentation_intel
```

[//]: # (capture: baremetal)
```bash
rm -rf $DATA_DIR/Online* $OUTPUT_DIR
```

To remove WORKSPACE:

[//]: # (capture: baremetal)
```bash
rm -rf $WORKSPACE
```

## Expected Output

When running `hyperparameter_cluster_analysis.py` with k-means as clustering algorithm, the output should show similar results as shown below:

```text
...
2023-11-29 23:32:38,561 [INFO] sklearn.utils.validation._assert_all_finite: running accelerated version on CPU
2023-11-29 23:32:38,820 [INFO] Model saved to /workspace/output/models/kmeans/model_30_1e-05.pkl
2023-11-29 23:32:38,820 [INFO] Predictions saved to /workspace/output/models/kmeans/pred_30_1e-05.txt
2023-11-29 23:32:38,820 [INFO] ===> Training Time (clusters = 30, tol = 0.000010) : 0.256 secs
2023-11-29 23:32:38,891 [INFO] Total hyperparameter tuning time : 4.203 secs
```

When using DBSCAN as clustering algorithm, the output should show similar results as shown below:

```text
...
2023-11-29 23:40:00,176 [INFO] sklearn.utils.validation._assert_all_finite: running accelerated version on CPU
2023-11-29 23:40:00,194 [INFO] Model saved to /workspace/output/models/dbscan/model_100_0.7.pkl
2023-11-29 23:40:00,194 [INFO] Predictions saved to /workspace/output/models/dbscan/pred_100_0.7.txt
2023-11-29 23:40:00,194 [INFO] ===> Training Time (min_samples = 100, eps = 0.700000) : 0.017 secs
2023-11-29 23:40:00,242 [INFO] Total hyperparameter tuning time : 0.273 secs
```

## Summary and Next Steps

So far a clustering model was trained for customer segmentation. If user wants to use the model inside a script, two examples are provided:

* The following example can be used to load the saved model files and predictions for further analysis:

```python
import os
import joblib
import pandas as pd

OUTPUT_DIR = os.getenv('OUTPUT_DIR')
hyperparameters = "30_1e-05"
algo = "kmeans"

model = joblib.load(f"{OUTPUT_DIR}/models/{algo}/model_{hyperparameters}.pkl")
data = pd.read_csv(f"{OUTPUT_DIR}/models/{algo}/data.csv")
cluster_labels = pd.read_csv(f"{OUTPUT_DIR}/models/{algo}/pred_{hyperparameters}.txt", header=None)
```

* The saved model can be loaded using the `joblib` module and used to predict the cluster label of a new data point. This may look like:

```python
import os
import joblib
import pandas as pd

OUTPUT_DIR = os.getenv('OUTPUT_DIR')
hyperparameters = "30_1e-05"
algo = "kmeans"

columns = ['Amount', 'NumberOfOrders', 'Recency', 'Friday', 'Monday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday', 'April', 'August', 'December', 'February', 'January', 'July', 'June', 'March', 'May', 'November', 'October', 'September']

value = [[5.0, 2.0, 325.0, 20.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 500.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

new_X=pd.DataFrame(value, columns=columns)

kmeans_model = joblib.load(f"{OUTPUT_DIR}/models/{algo}/model_{hyperparameters}.pkl")
result = kmeans_model.predict(new_X)
print(result)
```

## Learn More

For more information about or to read about other relevant workflow examples, see these guides and software resources:

* [Intel® AI Analytics Toolkit (AI Kit)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html)
* [Intel® Distribution for Python*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-Python*.html)
* [Intel® Distribution of Modin*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-of-modin.html)
* [Intel® Extension for Scikit-Learn*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/scikit-learn.html)

## Support

The End-to-end Predictive Asset Health Analytics team tracks both bugs and
enhancement requests using [GitHub
issues](https://github.com/oneapi-src/customer-segmentation/issues).
Before submitting a suggestion or bug report,
see if your issue has already been reported.

## Appendix

### Disclaimers

Performance varies by use, configuration and other factors. Learn more on the [Performance Index site](https://edc.intel.com/content/www/us/en/products/performance/benchmarks/overview/).</br>
Performance results are based on testing as of dates shown in configurations and may not reflect all publicly available updates.  See backup for configuration details.  No product or component can be absolutely secure. </br>
Your costs and results may vary. </br>
Intel technologies may require enabled hardware, software or service activation. </br>
© Intel Corporation.  Intel, the Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries.  Other names and brands may be claimed as the property of others. </br>

To the extent that any public or non-Intel datasets or models are referenced by or accessed using tools or code on this site those datasets or models are provided by the third party indicated as the content source. Intel does not create the content and does not warrant its accuracy or quality. By accessing the public content, or using materials trained on or with such content, you agree to the terms associated with that content and that your use complies with the applicable license.

Intel expressly disclaims the accuracy, adequacy, or completeness of any such public content, and is not liable for any errors, omissions, or defects in the content, or for any reliance on the content. Intel is not liable for any liability or damages relating to your use of public content.

\*Names and brands that may be claimed as the property of others. [Trademarks](https://www.intel.com/content/www/us/en/legal/trademarks.html).