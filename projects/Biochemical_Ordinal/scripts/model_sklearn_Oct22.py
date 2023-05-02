import pathlib

from sklearn.utils.extmath import softmax

import dataset_Oct22
import model_config_Oct22


model_names = ["SklearnRidgeRegression", "SklearnLogisticRegression"]

def add_common_arguments(parser, model_names = model_names):
    group = parser.add_argument_group("general")
    group.add_argument("-m", "--model_name",
        help="Model name", default=model_names[0], choices=model_names)
    group.add_argument("-i", "--input",
        help="Oct22 sequences csv file", type=pathlib.Path,
        default="../output/ordinal_Oct22_sequences_with_dca_score.csv")
    group.add_argument("-o", "--output_dir",
        help="Location to store output files", 
        default="../output/Oct22_models", type=pathlib.Path)
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
                       choices=['multiclass", "binary'], type=str)
    group.add_argument("--dca", help="Add DCA score", action="store_true")
    return group

def add_sklearn_design_arguments(group):
    group.add_argument("--intercept", help="Add intercept to design matrix",
                       action="store_true")
    return group

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
    logging.info(str(ds).replace("\n", ", "))

    mc = model_config_Oct22.ModelConfig(
            model_name = args.model_name,
            target = args.target,
            intercept = args.intercept,
            dca = args.dca,
            encoding = args.encoding
            )
    logging.info(str(mc).replace("\n", ", "))



