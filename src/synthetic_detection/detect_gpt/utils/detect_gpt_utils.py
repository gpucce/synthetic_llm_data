
import numpy as np


def compute_prob(x, axis=None):
    return np.sum(np.log(x), axis=axis) / len(x)


def detect_gpt(df, true_col):
    df_true = df.loc[:, true_col].apply(compute_prob).values
    df_syn = (
        df.loc[:, [i for i in df.columns if i != true_col]].applymap(compute_prob).values
    )

    mean_df_syn = np.mean(df_syn, axis=1)
    unnorm_denom = df_syn - mean_df_syn.reshape(-1, 1)
    denom = np.sqrt(np.power(unnorm_denom, 2).sum(axis=1) / (df_syn.shape[1] - 1))
    return (df_true - mean_df_syn) / denom
