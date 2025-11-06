import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn import metrics as skm
from sklearn.preprocessing import scale, minmax_scale


def standardized_normalized_ndcg(true_scores, predicted_scores, k=None):
    """ NDCG metric, but the true scores are standardized (z-scores) to help comparisons across datasets
        Also, the z-scores are min-max normalized to the range between 0 and 1 because NDCG does not
        handle negative values correctly

        source for standardization: Listgarten paper
        source for min-max normalization: https://dl.acm.org/doi/abs/10.1145/3340531.3412123 """

    # compute z-scores
    standardized_true_scores = scale(true_scores)
    normalized_true_scores = minmax_scale(standardized_true_scores)
    return skm.ndcg_score(y_true=np.expand_dims(normalized_true_scores, axis=0),
                          y_score=np.expand_dims(predicted_scores, axis=0),
                          k=k)


def compute_metrics(targets, predictions,
                    metrics=("mse", "pearsonr", "r2", "spearmanr")):
    metrics_dict = {}
    for metric in metrics:
        if np.isnan(targets).any() or np.isnan(predictions).any():
            metrics_dict[metric] = np.nan
        elif metric == "mse":
            metrics_dict["mse"] = skm.mean_squared_error(targets, predictions)
        elif metric == "pearsonr":
            metrics_dict["pearsonr"] = pearsonr(targets, predictions)[0]
        elif metric == "spearmanr":
            metrics_dict["spearmanr"] = spearmanr(targets, predictions)[0]
        elif metric == "r2":
            metrics_dict["r2"] = skm.r2_score(targets, predictions)
        elif metric == "ndcg":
            metrics_dict["ndcg"] = standardized_normalized_ndcg(targets, predictions)
        elif metric == "ndcg_100":
            metrics_dict["ndcg_100"] = standardized_normalized_ndcg(targets, predictions, k=100)

    return metrics_dict
