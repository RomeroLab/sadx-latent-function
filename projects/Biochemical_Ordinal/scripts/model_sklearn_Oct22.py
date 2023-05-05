import pathlib

from sklearn.utils.extmath import softmax
from scipy.special import expit

import dataset_Oct22
import model_config_Oct22

from sklearn.linear_model import LogisticRegressionCV, RidgeClassifierCV


model_names = ["SklearnRidgeClassifier", "SklearnLogisticRegression"]

def add_common_arguments(parser, model_names = model_names):
    group = parser.add_argument_group("general")
    group.add_argument("-m", "--model_name",
        help="Model name", default=model_names[0], choices=model_names)
    group.add_argument("-i", "--input",
        help="Oct22 sequences csv file", type=pathlib.Path,
        default="../output/ordinal_Oct22_sequences_with_dca_score.csv")
    group.add_argument("-o", "--output_dir",
        help="Location to store output files", 
        default="../output/ordinal_Oct22_models", type=pathlib.Path)
    group.add_argument("-l", "--loglevel",
        help="Logging Level", default="INFO")
    return group

def add_design_arguments(parser):
    group = parser.add_argument_group("design")
    # only one-hot supported for sklearn models
    group.add_argument("-e", "--encoding",
        help="Encoding", default="one-hot", choices=['one-hot'],
                       type=str)
    group.add_argument("-t", "--target", 
        help="target variable", default="multiclass", 
                       choices=["multiclass", "binary"], type=str)
    group.add_argument("--dca", help="Add DCA score", action="store_true")
    return group

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


def get_model_data(
        model_config, 
        dataset, 
        # data type should be train or test
        data_type="train" ):
    # get train or test data
    model_data = None
    if data_type == "train":
        model_data = dataset.get_train_dataset()
    elif data_type == "test":
        model_data = dataset.get_test_dataset()
    else:
        raise ValueError("data_type should be 'train' or 'test'")

    X = dataset_Oct22.create_model_inputs(model_data, 
        add_dca=model_config.design_matrix["dca"])
    y = dataset_Oct22.create_target(model_data, 
        activity_only = True if model_config.target == "binary" else False)
    return X, y


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

    mc = model_config_Oct22.ModelConfig(
            model_name = args.model_name,
            target = args.target,
            intercept = args.intercept,
            dca = args.dca,
            encoding = args.encoding
            )
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
    # we do this to look at AUC curves
    des = model.decision_function(X_test)

    probs = None
    if mc.target == "binary":
        assert(len(des.shape) == 1)
        p = expit(des) # probablity of predicting 1.
        probs = np.vstack([1-p, p])
    elif mc.target == "multiclass":
        assert(des.shape[1] == 4)
    else:
        raise ValueError(f"Got unknown value of mc.target={mc.target}")

