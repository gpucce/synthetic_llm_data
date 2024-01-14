
import matplotlib.pyplot as plt
from sklearn.metrics import auc


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
    return fig, ax
