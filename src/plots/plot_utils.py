
from pathlib import Path
from datasets import load_dataset, load_from_disk
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt


def plot_roc(fpr, tpr, ax=None, label=None):
    roc_auc = auc(fpr, tpr)
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=100)
    label_string = f'AUC = {roc_auc:0.2f}'
    if label is not None:
        label_string = label + label_string
    ax.plot(fpr, tpr, label = label_string)
    ax.legend(loc = 'lower right')
    ax.plot([0, 1], [0, 1],'r--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    return fig, ax, roc_auc

def plot_roc_from_dataset(data_path, col_name, output_path):
    output_path = Path(output_path)
    ds = load_from_disk(data_path)

    fpr_llh, tpr_llh, _ = roc_curve(ds["label"], ds[f"{col_name}_loss"])
    fig_llh, ax_llh, llh_auroc = plot_roc(fpr_llh, tpr_llh, label="loglikelihood ")

    fpr, tpr, _ = roc_curve(ds["label"], ds[f"{col_name}_score"])
    _, _, detectgpt_auroc = plot_roc(fpr, tpr, ax=ax_llh, label="detectGPT ")

    fig_llh.savefig(output_path / "roc_curve.png")
    return fig_llh, detectgpt_auroc, llh_auroc
