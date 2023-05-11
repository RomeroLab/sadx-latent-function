import pathlib
import argparse


import dataset_Oct22

GENERIC_MODEL_NAMES = ["model"]

def add_common_arguments(parser, model_names = GENERIC_MODEL_NAMES):
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


