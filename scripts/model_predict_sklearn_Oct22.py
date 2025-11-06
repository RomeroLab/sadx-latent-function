import pathlib

import numpy as np
from sklearn.utils.extmath import softmax
from scipy.special import expit

import dataset_Oct22
import model_config_Oct22

from sklearn.linear_model import LogisticRegressionCV, RidgeClassifierCV

from model_shared_Oct22 import  \
        add_common_arguments, add_design_arguments, get_model_data

model_names = ["SklearnRidgeClassifier", "SklearnLogisticRegression"]

import torch
from torch.utils.data import Dataset, DataLoader

import utils
utils.add_projects_to_path()
from protein_utils.sequences.encoding import DEFAULT_NO_GAP_ENCODER
from protein_utils.energy.potts import energy_calc_single_mutants, create_single_mutant


from model_predict_pytorch_Oct22 import SingleMutant_Oct22DataSet

if __name__ == "__main__":
    import sys
    import argparse
    import logging

    parser = argparse.ArgumentParser()
    model_names = ["SklearnLogisticRegression"]
    group = add_common_arguments(parser, model_names = model_names)
    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout,
                            level=getattr(logging, args.loglevel))
    

    mc = model_config_Oct22.ModelConfig.create_from_yaml(
            "../output/ordinal_Oct22_models/6d97d935.yml")
    logging.info("model_config : " + str(mc).replace("\n", ", "))
    logging.info(f"Creating dataset")
    ds = dataset_Oct22.Oct22DataSet(args.input)
    logging.info("dataset : " + str(ds).replace("\n", ", "))

    assert(mc.model_name == "SklearnLogisticRegression")
    model = LogisticRegressionCV(
        fit_intercept = mc.design_matrix["intercept"],
        random_state = mc.seed,
        max_iter=1000,
        cv=None)
  
    X_train, y_train = get_model_data(model_config=mc, dataset=ds, 
                                        data_type = "all")
    logging.info(f"X_train.shape = {X_train.shape}, "
                 f"y_train.shape = {y_train.shape}")

    # train the model
    clf = model.fit(X_train, y_train)

    predict_dl = DataLoader(SingleMutant_Oct22DataSet(multilibrary=True), 
            batch_size=32, num_workers=4)
 
    predictions = [clf.predict(batch["encoding"].numpy()) for batch in predict_dl]
    predictions = torch.hstack([torch.Tensor(p) for p in predictions]).numpy()

    probas = [clf.predict_proba(batch["encoding"].numpy()) for batch in predict_dl]
    probas = torch.cat([torch.Tensor(p) for p in probas], dim=0).numpy()


