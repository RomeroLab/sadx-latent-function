"""
Evaluate models on the test set using Mean Absolute Error (MAE).

Reads model configurations directly from their yaml files rather than
hardcoding values. Produces a two-panel figure: a summary table and
a bar chart of MAE performance.

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
import matplotlib
import matplotlib.pyplot as plt
import yaml


def setup_logging(output_dir):
    """Configure logging to both terminal and file."""
    logger = logging.getLogger("eval_mae")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s — %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

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


def load_model_yaml(models_dir, uuid):
    """Load a model's yaml config file."""
    yml_file = pathlib.Path(models_dir) / f"{uuid}.yml"
    with open(yml_file, "r") as f:
        return yaml.safe_load(f)


def compute_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def bool_to_yesno(val):
    """Convert a boolean or truthy value to Yes/No."""
    if isinstance(val, bool):
        return "Yes" if val else "No"
    if isinstance(val, str):
        return "Yes" if val.lower() == "true" else "No"
    return "No"


def parse_model_properties(model_yaml, display_names):
    """Extract display-ready properties from a model yaml."""
    model_name_raw = model_yaml.get("model_name", "")
    model_type = display_names.get("model_name", {}).get(
        model_name_raw, model_name_raw)

    dm = model_yaml.get("design_matrix", {})
    tp = model_yaml.get("training_params", {})

    dca = bool_to_yesno(dm.get("dca", False))
    library_bias = bool_to_yesno(dm.get("multilibrary", False))

    is_sklearn = model_name_raw.startswith("Sklearn")

    props = {
        "type": model_type,
        "dca": dca,
        "library_bias": library_bias,
    }

    if is_sklearn:
        props["intercept"] = bool_to_yesno(dm.get("intercept", False))
        props["epochs"] = "\u2014"
        props["lr"] = "\u2014"
        props["wd"] = "\u2014"
    else:
        props["intercept"] = "\u2014"
        props["epochs"] = str(tp.get("num_epochs", "\u2014"))
        props["lr"] = f"{tp.get('learning_rate', 0):.0e}"
        props["wd"] = f"{tp.get('weight_decay', 0):.0e}"

    return props


def main(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    paths = config["paths"]
    display_names = config.get("display_names", {})

    output_dir = create_output_dir(paths["output_base_dir"])
    shutil.copy2(config_file, output_dir / "eval_config.yml")

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
    model_props_list = []

    # evaluate each model
    for model in config["models"]:
        label = model["label"]
        uuid = model["uuid"]
        selected = model.get("selected", False)

        y_pred = load_predictions(paths["models_dir"], uuid)
        assert len(y_pred) == n_test, \
            f"{label}: expected {n_test} predictions, got {len(y_pred)}"
        mae = compute_mae(y_true, y_pred)

        model_yaml = load_model_yaml(paths["models_dir"], uuid)
        props = parse_model_properties(model_yaml, display_names)

        results.append({
            "label": label, "uuid": uuid,
            "mae": mae, "selected": selected,
        })
        model_props_list.append({
            "label": label, "selected": selected, **props, "mae": mae,
        })

        logger.info(f"{label:30s} (uuid={uuid}): MAE = {mae:.4f}")
        logger.info(f"  type={props['type']}, dca={props['dca']}, "
                     f"library_bias={props['library_bias']}, "
                     f"epochs={props['epochs']}, lr={props['lr']}, wd={props['wd']}")

    # evaluate baselines
    for baseline in config["baselines"]:
        label = baseline["label"]
        value = baseline["value"]
        y_pred = np.full(n_test, value)
        mae = compute_mae(y_true, y_pred)
        results.append({
            "label": label, "uuid": "baseline",
            "mae": mae, "selected": False,
        })
        model_props_list.append({
            "label": label, "selected": False,
            "type": "Baseline", "dca": "\u2014", "library_bias": "\u2014",
            "intercept": "\u2014", "epochs": "\u2014", "lr": "\u2014",
            "wd": "\u2014", "mae": mae,
        })
        logger.info(f"{label:30s} (constant={value}): MAE = {mae:.4f}")

    # sort by MAE (lowest first)
    results.sort(key=lambda x: x["mae"])

    # save raw values as csv
    results_df = pd.DataFrame(model_props_list)
    results_csv = output_dir / "mae_results.csv"
    results_df.to_csv(results_csv, index=False)
    logger.info(f"Raw results saved to {results_csv}")

    # ---- FIGURE ----
    labels_sorted = [r["label"] for r in results]
    maes_sorted = [r["mae"] for r in results]

    # font setup - use DejaVu Serif as fallback if Times not available
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 12,
        "mathtext.fontset": "dejavuserif",
    })

    fig = plt.figure(figsize=(20, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1], wspace=0.15)
    ax_table = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])

    # --- LEFT PANEL: model summary table ---
    ax_table.axis("off")

    table_headers = [
        "Label", "Model Type", "DCA",
        "Library-Specific\nBias Terms",
        "Epochs", "Learning\nRate", "Weight\nDecay",
    ]
    table_data = []
    row_selected = []
    row_is_baseline = []
    for mp in model_props_list:
        label_display = mp["label"]
        if mp.get("selected", False):
            label_display = f"{label_display}  *"
        table_data.append([
            label_display,
            mp["type"],
            mp["dca"],
            mp["library_bias"],
            mp["epochs"],
            mp["lr"],
            mp["wd"],
        ])
        row_selected.append(mp.get("selected", False))
        row_is_baseline.append(mp["type"] == "Baseline")

    table = ax_table.table(
        cellText=table_data,
        colLabels=table_headers,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)
    table.auto_set_column_width(col=list(range(len(table_headers))))

    # style header row
    for j in range(len(table_headers)):
        cell = table[0, j]
        cell.set_facecolor("#2C3E50")
        cell.set_text_props(color="white", fontweight="bold", fontsize=10)
        cell.set_edgecolor("#1A252F")
        cell.set_height(0.12)

    # style data rows
    for i in range(len(table_data)):
        for j in range(len(table_headers)):
            cell = table[i + 1, j]
            cell.set_edgecolor("#D5D8DC")
            if row_selected[i]:
                cell.set_facecolor("#85C1E9")
                cell.set_text_props(fontweight="bold", fontsize=11)
            elif row_is_baseline[i]:
                cell.set_facecolor("#E8E8E8")
                cell.set_text_props(fontsize=11, color="#2C3E50",
                                    fontstyle="italic")
            else:
                # alternate row shading
                bg = "#FFFFFF" if i % 2 == 0 else "#F8F9F9"
                cell.set_facecolor(bg)
                cell.set_text_props(fontsize=11)

    ax_table.set_title("Model Configurations", fontsize=15,
                       fontweight="bold", pad=25)

    # --- RIGHT PANEL: bar chart ---
    def get_color(label):
        if label.startswith("Always"):
            return "#BDC3C7"
        elif label == "MLP":
            return "#2471A3"
        elif label.startswith("MLP"):
            return "#5DADE2"
        elif label.startswith("Linear"):
            return "#DC7633"
        elif label.startswith("Ridge"):
            return "#27AE60"
        return "#95A5A6"

    colors = [get_color(l) for l in labels_sorted]

    bars = ax_bar.barh(range(len(labels_sorted)), maes_sorted,
                       color=colors, edgecolor="#2C3E50", linewidth=0.8,
                       height=0.7)
    ax_bar.set_yticks(range(len(labels_sorted)))
    ax_bar.set_yticklabels(labels_sorted, fontsize=12)
    ax_bar.set_xlabel("Mean Absolute Error (MAE)", fontsize=13)
    ax_bar.set_title("Test Set Performance: Mean Absolute Error (lower is better)",
                     fontsize=15, fontweight="bold", pad=25)
    ax_bar.tick_params(axis='x', labelsize=11)
    ax_bar.invert_yaxis()
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    # highlight selected model
    selected_labels = {m["label"] for m in config["models"]
                       if m.get("selected", False)}
    for i, label in enumerate(labels_sorted):
        if label in selected_labels:
            bars[i].set_linewidth(2.5)
            bars[i].set_edgecolor("#1A5276")

    # add MAE values with enough room
    max_mae = max(maes_sorted)
    ax_bar.set_xlim(0, max_mae * 1.2)

    for i, (bar, mae) in enumerate(zip(bars, maes_sorted)):
        label = labels_sorted[i]
        mae_text = f"{mae:.4f}"
        if label in selected_labels:
            mae_text = f"{mae:.4f}  (selected)"
        ax_bar.text(bar.get_width() + max_mae * 0.015,
                    bar.get_y() + bar.get_height() / 2,
                    mae_text, va="center", fontsize=11,
                    fontweight="bold" if label in selected_labels else "normal")

    figure_path = output_dir / "mae_comparison.png"
    plt.savefig(figure_path, dpi=150, bbox_inches="tight",
                facecolor="white", pad_inches=0.3)
    logger.info(f"Figure saved to {figure_path}")
    plt.close()

    logger.info("Done")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python eval_mae.py eval_config.yml")
        sys.exit(1)
    main(sys.argv[1])