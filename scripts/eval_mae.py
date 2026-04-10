"""
Evaluate models on the test set using Mean Absolute Error (MAE).

Usage:
    python eval_mae.py eval_config.yml
"""

import sys
import shutil
import pathlib
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml


def setup_logging(output_dir):
    """Configure logging to both terminal and file."""
    logger = logging.getLogger("eval_mae")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s — %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")

    # terminal
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # file
    log_file = output_dir / "eval_mae.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def create_output_dir(base_dir):
    """Create a timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = pathlib.Path(base_dir) / f"model_evals_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_true_labels(data_csv, test_indices_file):
    """Load true category labels for the test set."""
    df = pd.read_csv(data_csv)
    category_map = {'N': 0, 'L': 1, 'P': 2, 'H': 3}
    df["category_num"] = df["category"].map(category_map)

    test_indices = np.loadtxt(test_indices_file, dtype=int)
    y_true = df.iloc[test_indices]["category_num"].to_numpy()
    return y_true


def load_predictions(models_dir, uuid):
    """Load predictions from a preds.txt file."""
    preds_file = pathlib.Path(models_dir) / f"{uuid}.preds.txt"
    preds = np.loadtxt(preds_file)
    return preds.astype(int)


def compute_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def main(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    paths = config["paths"]

    # create timestamped output directory
    output_dir = create_output_dir(paths["output_base_dir"])

    # save a copy of the config
    shutil.copy2(config_file, output_dir / "eval_config.yml")

    # setup logging
    logger = setup_logging(output_dir)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Config file: {config_file}")

    # load true labels
    y_true = load_true_labels(paths["data_csv"], paths["test_indices"])
    n_test = len(y_true)
    logger.info(f"Loaded {n_test} test samples")
    unique, counts = np.unique(y_true, return_counts=True)
    label_names = {0: 'N', 1: 'L', 2: 'P', 3: 'H'}
    dist_str = ", ".join(f"{label_names[u]}={c}" for u, c in zip(unique, counts))
    logger.info(f"True label distribution: {dist_str}")

    results = []

    # evaluate each model
    for model in config["models"]:
        label = model["label"]
        uuid = model["uuid"]
        y_pred = load_predictions(paths["models_dir"], uuid)
        assert len(y_pred) == n_test, \
            f"{label}: expected {n_test} predictions, got {len(y_pred)}"
        mae = compute_mae(y_true, y_pred)
        results.append({"label": label, "uuid": uuid, "mae": mae})
        logger.info(f"{label:30s} (uuid={uuid}): MAE = {mae:.4f}")

    # evaluate baselines
    for baseline in config["baselines"]:
        label = baseline["label"]
        value = baseline["value"]
        y_pred = np.full(n_test, value)
        mae = compute_mae(y_true, y_pred)
        results.append({"label": label, "uuid": "baseline", "mae": mae})
        logger.info(f"{label:30s} (constant={value}): MAE = {mae:.4f}")

    # sort by MAE (lowest first)
    results.sort(key=lambda x: x["mae"])

    # save raw values as csv
    results_df = pd.DataFrame(results)
    results_csv = output_dir / "mae_results.csv"
    results_df.to_csv(results_csv, index=False)
    logger.info(f"Raw results saved to {results_csv}")

    # plot
    labels = [r["label"] for r in results]
    maes = [r["mae"] for r in results]

    # font setup
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 12,
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = []
    for label in labels:
        if label.startswith("Always"):
            colors.append("#B0BEC5")   # cool gray
        elif label == "MLP":
            colors.append("#1565C0")   # deep blue
        elif label.startswith("MLP"):
            colors.append("#5C9BD4")   # medium blue
        elif label.startswith("Linear"):
            colors.append("#D4813A")   # warm amber
        elif label.startswith("Ridge"):
            colors.append("#6A994E")   # sage green
        else:
            colors.append("#9E9E9E")

    bars = ax.barh(range(len(labels)), maes, color=colors,
                   edgecolor="#2C2C2C", linewidth=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("Mean Absolute Error (MAE)", fontsize=14)
    ax.set_title("Model Comparison on Test Set (lower is better)", fontsize=16)
    ax.tick_params(axis='x', labelsize=11)
    ax.invert_yaxis()

    for bar, mae in zip(bars, maes):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{mae:.2f}", va="center", fontsize=11)

    plt.tight_layout()
    figure_path = output_dir / "mae_comparison.png"
    plt.savefig(figure_path, dpi=150)
    logger.info(f"Figure saved to {figure_path}")
    plt.close()

    logger.info("Done")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python eval_mae.py eval_config.yml")
        sys.exit(1)
    main(sys.argv[1])