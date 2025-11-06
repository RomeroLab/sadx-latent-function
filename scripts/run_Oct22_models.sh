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

python3 model_pytorch_Oct22.py \
        --train_name train --val_name val_cv1 --test_name test \
        --num_epochs  50  --lambda_h 0.0  --learning_rate 1e-2  --weight_decay 1e-4 \
        --target multiclass  --dca --multilibrary \
        --save_model
