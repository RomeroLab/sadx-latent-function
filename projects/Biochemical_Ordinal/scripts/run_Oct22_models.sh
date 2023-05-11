#!/bin/bash


# Ridge Classifier
python3 model_sklearn_Oct22.py -m SklearnRidgeClassifier -t binary
python3 model_sklearn_Oct22.py -m SklearnRidgeClassifier -t binary --intercept
python3 model_sklearn_Oct22.py -m SklearnRidgeClassifier -t multiclass
python3 model_sklearn_Oct22.py -m SklearnRidgeClassifier -t multiclass --intercept

# Logistic Regression
python3 model_sklearn_Oct22.py -m SklearnLogisticRegression -t binary
python3 model_sklearn_Oct22.py -m SklearnLogisticRegression -t binary --intercept
python3 model_sklearn_Oct22.py -m SklearnLogisticRegression -t multiclass
python3 model_sklearn_Oct22.py -m SklearnLogisticRegression -t multiclass --intercept
