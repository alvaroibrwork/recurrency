# Unveiling Recurring Financial Patterns: Novel unsupervised filtering algorithms for enhanced forecasting

This directory contains the code for reproducing the experiments of the paper "Unveiling Recurring Financial Patterns: Novel unsupervised filtering algorithms for enhanced forecasting".

The code for the experiments is organized in several notebooks under the `experiments/` directory.


## Content of the Experiments directory

The experiments are organized in a notebook format so it's easier to interact with the intermediate results.

**experiments/00_bank**

Runs the filtering algorithms on  the original BANK dataset. It has three special cells – one for each method – which filter the dataset with the desired method.

**experiments/00_berka**

Runs the filtering algorithms on  the original BERKA dataset. It has three special cells – one for each method – which filter the dataset with the desired method.

**experiments/01_Prediction**

Takes the outputs produced by the '00_berka' notebook and runs the test of training a model on the unfiltered dataset, getting its metrics and repeating the same with a filtered version

**experiments/02_Assessment**

Builds corrupted versions of the dataset (artificial noise) and runs the filtering using all methods in order to benchmark them.

**experiments/03_Performance_comparison**

Measures the scalability of each algorithm in terms of complexity and noise tolerance.


## How to run

Install the requirements

```
pip install -r requirements.txt
```

Place the datasets in the `datasets` directory:

*BERKA*:
Download the BERKA dataset from [Kaggle](https://www.kaggle.com/datasets/marceloventura/the-berka-dataset) and place the file `trans.asc` inside the `datasets/` directory.

*BANK*:
Request access to the BANK and BANK AUTOMATED datasets and place the two parquet files inside the `datasets/` directory.

For requesting access you must:
1) Go to [Zenodo using this link](https://zenodo.org/records/11121973).
2) Request access to the data.
3) Accept the requirements of not re-sharing it without permission of the authors (data will be freely available in the future, after the review process).
4) Download!
5) Place the `.parquet` files in the corresponding directory



Run a jupyter notebook server to see and interact with our notebooks.

```
jupyter lab
```
