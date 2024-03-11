
from pathlib import Path
from datasets import load_from_disk
from sklearn.metrics import auc, roc_curve, classification_report
import matplotlib.pyplot as plt


def plot_roc(fpr, tpr, ax=None, label=None):
    roc_auc = auc(fpr, tpr)
    plt.tight_layout()
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=100)
    label_string = f'AUC = {roc_auc:0.2f}'
    if label is not None:
        label_string = label + label_string
    ax.plot(fpr, tpr, label = label_string)
    ax.legend(loc = 'lower right', fontsize=12)
    ax.plot([0, 1], [0, 1],'r--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_ylabel('True Positive Rate', fontsize=20)
    ax.set_xlabel('False Positive Rate', fontsize=20)
    return fig, ax, roc_auc

def plot_roc_from_dataset(data_path, col_name, output_path):
    output_path = Path(output_path)
    ds = load_from_disk(data_path)

    print(data_path)
    def normalize(x, thr):
        _min, _max = min(x), max(x)
        return [((i - _min) / (_max - _min)) > thr for i in x]
    print(classification_report(ds["label"], normalize(ds[f"{col_name}_score"], 0.35)))

    fpr_llh, tpr_llh, _ = roc_curve(ds["label"], ds[f"{col_name}_loss"])
    fig_llh, ax_llh, llh_auroc = plot_roc(fpr_llh, tpr_llh, label="loglikelihood ")

    fpr, tpr, _ = roc_curve(ds["label"], ds[f"{col_name}_score"])
    _, _, detectgpt_auroc = plot_roc(fpr, tpr, ax=ax_llh, label="detectGPT ")

    fig_llh.savefig(output_path / "roc_curve.png")
    return fig_llh, detectgpt_auroc, llh_auroc

def plot_supervised_accuracy(ds):
    ds = ds.groupby("mixed_data")
    figs = []
    for data_mix in ["in_domain", "mixed"]:
        for detector in ["roberta-large", "xlm-roberta-large"]:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            plt.tight_layout()
            (
                ds.get_group(data_mix).drop("mixed_data", axis=1)
                .pivot(index="n samples", columns="generator", values=detector)
                .plot(kind="bar", ax=ax)
            )
            # Set xticks label size
            ax.set_ylabel("Accuracy", fontsize=20)
            ax.set_xlabel("Number of samples", fontsize=20)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=15, rotation=0)

            # Set the legend above the plot
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2, fontsize=15)

            figs.append(fig)
    
    return figs