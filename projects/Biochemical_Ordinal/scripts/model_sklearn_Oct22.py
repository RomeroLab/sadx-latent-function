import pathlib

import numpy as np
from sklearn.utils.extmath import softmax
from scipy.special import expit

import dataset_Oct22
import model_config_Oct22

from sklearn.linear_model import LogisticRegressionCV, RidgeClassifierCV

from model_shared_Oct22 import model_names, \
        add_common_arguments, add_design_arguments, get_model_data



def add_sklearn_design_arguments(group):
    group.add_argument("--intercept", help="Add intercept to design matrix",
                       action="store_true")
    return group

def create_model(model_config, cv=None):
    """ Takes a model config object and returns a model that can run """
    model = None
    if model_config.model_name == "SklearnLogisticRegression":
        model = LogisticRegressionCV(
                    fit_intercept = model_config.design_matrix["intercept"],
                    random_state = model_config.seed,
                    max_iter=1000,
                    cv=cv)
    elif model_config.model_name == "SklearnRidgeClassifier":
        model = RidgeClassifierCV(
                    fit_intercept = model_config.design_matrix["intercept"],
                    cv = cv)
    else:
        raise ValueError(f"{model_name} must be one of {model_names}")
    return model



if __name__ == "__main__":
    import sys
    import argparse
    import logging

    parser = argparse.ArgumentParser()
    group = add_common_arguments(parser)
    group = add_design_arguments(parser)
    group = add_sklearn_design_arguments(parser)
    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout,
                            level=getattr(logging, args.loglevel))
    
    logging.info(f"Reading filename: {args.input}")
    ds = dataset_Oct22.Oct22DataSet(args.input)
    logging.info("dataset : " + str(ds).replace("\n", ", "))

    mc = model_config_Oct22.ModelConfig.create_from_args(args)

    logging.info("model_config : " + str(mc).replace("\n", ", "))
    mc.save_yaml(directory=args.output_dir)

    # the dataset object can be used as cv as it iterates over the 
    # split training indices
    model = create_model(model_config = mc, cv = ds)

    # get the training data
    X_train, y_train = get_model_data(model_config=mc, dataset=ds, 
                                        data_type = "train")
    logging.info(f"X_train.shape = {X_train.shape}, "
                 f"y_train.shape = {y_train.shape}")

    # train the model
    clf = model.fit(X_train, y_train)

    # get testing data
    X_test, y_test = get_model_data(model_config=mc, dataset=ds, 
                                    data_type = "test")

    logging.info(f"model_score = {model.score(X_test, y_test)}")

    # convert decision scores to probabilities
    # we do this to look at AUC curves for the binary targets
    des = model.decision_function(X_test)

    probs = None
    argmax_axis = 0
    if mc.target == "binary":
        assert(len(des.shape) == 1)
        p = expit(des) # probablity of predicting 1.
        probs = np.vstack([1-p, p])
        argmax_axis = 0
    elif mc.target == "multiclass":
        assert(des.shape[1] == 4)
        probs = softmax(des)
        argmax_axis = 1
    else:
        raise ValueError(f"Got unknown value of mc.target={mc.target}")
    
    logging.info(f"Saving probabilities")
    np.savetxt(args.output_dir / f"{mc.uuid}.probs.txt", probs)

    model_test_predictions = model.predict(X_test)
    np.savetxt(args.output_dir / f"{mc.uuid}.preds.txt", model_test_predictions)
    if len(np.unique(model_test_predictions)) == 1:
        logging.warning(f"~~ ALERT! all identical predictions on test set !! ~~")

    # check that the maximum probability is the same as a predict
    assert((probs.argmax(axis=argmax_axis) == model_test_predictions).all())
     
