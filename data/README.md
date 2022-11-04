## Setting up the data

The benchmarking scripts expects the files `data/OnlineRetail.csv`.

To setup the data for benchmarking under these requirements, do the following:

1. Download the data file "OnlineRetail.xlsx" from https://archive-beta.ics.uci.edu/ml/datasets/online+retail
2. Export the dataset as a csv and save it as `OnlineRetail.csv` in this directory.

```bash
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx
xlsx2csv Online\ Retail.xlsx OnlineRetail.csv
```

The `xlsx2csv` tool is used to convert the excel to a csv in the proper format.  On a Ubuntu machine, this can be installed as 

```bash
apt-get install xlsx2csv
```

> **Please see this data set's applicable license for terms and conditions. Intel Corporation does not own the rights to this data set and does not confer any rights to it.**