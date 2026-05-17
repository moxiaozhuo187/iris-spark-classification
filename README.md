# Iris Classification using Spark MLlib

## Overview

This project is for STQD6324 Data Management Assignment 1. The goal is 
to classify iris flowers into three species using three machine learning 
models built with Apache Spark MLlib.

I ran everything on Google Colab. Because I downloaded PySpark from Google Colab.One thing I adjusted during the process was the CV fold number.
Started with numFolds=3 but switched to 5 because 5-fold felt more solid 
for a dataset this small.

## Dataset

- Source: sklearn.datasets load_iris
- 150 samples, 4 features, 3 classes
- 50 samples each, perfectly balanced
- Features: sepal length, sepal width,
  petal length, petal width
- Classes: setosa, versicolor, virginica

## Methodology

Preprocessing:
- Checked for null values (none found)
- StringIndexer to convert species labels 
  to numeric: setosa=0.0, versicolor=1.0, 
  virginica=2.0
- VectorAssembler to combine 4 features 
  into one vector column
- 80/20 train test split with seed=1234

Models and hyperparameter search:

Logistic Regression
- regParam: [0.0, 0.1, 0.3]
- elasticNetParam: [0.0, 0.5, 1.0]
- maxIter: [20, 50]
- 18 combinations, 5-fold CV = 90 fits

Decision Tree
- maxDepth: [2, 3, 5]
- minInstancesPerNode: [1, 2, 3]
- 9 combinations, 5-fold CV = 45 fits

Random Forest
- numTrees: [10, 20, 50]
- maxDepth: [3, 5, 7]
- 6 combinations, 5-fold CV = 30 fits

Evaluation metrics used:
Accuracy, Precision, Recall, F1-score

## Results

| Model               | Accuracy | Precision | Recall | F1     |
|---------------------|----------|-----------|--------|--------|
| Logistic Regression | 0.9459   | 0.9543    | 0.9459 | 0.9471 |
| Decision Tree       | 0.9459   | 0.9543    | 0.9459 | 0.9463 |
| Random Forest       | 0.9459   | 0.9543    | 0.9459 | 0.9463 |

Best model: Logistic Regression (F1 = 0.9471)

## Key Findings

All three hit 0.9459. At first I thought something was wrong 
but it makes sense.37 test samples is not a lot. LR made 2 mistakes 
on the versicolor boundary.Since all three models hit the same accuracy,
they likely struggled with the same samples.

Feature importance from Random Forest:
- petal length: 0.4701
- petal width: 0.4312
- sepal length: 0.0973
- sepal width: 0.0013

Petal features carry over 90% of the importance.
Sepal width at 0.0013 contributes almost nothing.

The tuning part was the most useful for me.Seeing grid search pick 
the best params and then checking how CV score compared to test 
score was the main thing I took away.

## How to Run

1. Open Google Colab and connect the account
2. Cell 1 installs PySpark
3. Import libraries、start and initialize PySpark and load Iris dataset
4. Run all cells from top to bottom
5. For each steps performed, provide clear explanations and interpretations

## Requirements

- Google Colab
- PySpark (installed in cell 1)
- scikit-learn (pre-installed in Colab)
- pandas (pre-installed in Colab)

## Repository Structure

iris-spark-classification/
├── README.md
└── iris_classification.ipynb
