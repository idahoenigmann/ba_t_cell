import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from minimal import normalize, remove_outliers, separate, progress_bar, load_from_file_or_approx


def main(percentage, clustering_method):
    neg_df = load_from_file_or_approx(FILE_NAME_NEG)
    pos_df = load_from_file_or_approx(FILE_NAME_POS)
    neg_ctrl_df = load_from_file_or_approx(FILE_NAME_NEG_CTRL)
    pos_ctrl_df = load_from_file_or_approx(FILE_NAME_POS_CTRL)

    neg_df = remove_outliers(neg_df, 3, USED_COLUMNS)
    pos_df = remove_outliers(pos_df, 3, USED_COLUMNS)

    neg_idxs, pos_idxs = (sorted(set(neg_ctrl_df["idx"])),
                          sorted(set(pos_ctrl_df["idx"])))
    exp_neg_idxs = random.sample(neg_idxs, int(200 * percentage))
    exp_pos_idxs = random.sample(pos_idxs, int(200 * (1 - percentage)))
    exp_df = pd.concat([neg_ctrl_df[neg_ctrl_df["idx"].isin(exp_neg_idxs)],
                        pos_ctrl_df[pos_ctrl_df["idx"].isin(exp_pos_idxs)]])

    neg_df, pos_df, exp_df = normalize(neg_df, pos_df, exp_df, USED_COLUMNS)

    n = min(len(neg_df), len(pos_df))
    neg_df, pos_df = neg_df.sample(n), pos_df.sample(n)

    per, clustering = separate(neg_df, pos_df, USED_COLUMNS,
                               clustering_method)

    neg_df["predicted"] = clustering.predict(neg_df[USED_COLUMNS])
    pos_df["predicted"] = clustering.predict(pos_df[USED_COLUMNS])
    exp_df["predicted"] = clustering.predict(exp_df[USED_COLUMNS])

    TP = len(exp_df[(exp_df['predicted'] == per[1]) & (
        exp_df["idx"].isin(exp_pos_idxs))])
    FP = len(exp_df[(exp_df['predicted'] == per[1]) & (
        exp_df["idx"].isin(exp_neg_idxs))])
    TN = len(exp_df[(exp_df['predicted'] == per[0]) & (
        exp_df["idx"].isin(exp_neg_idxs))])
    FN = len(exp_df[(exp_df['predicted'] == per[0]) & (
        exp_df["idx"].isin(exp_pos_idxs))])

    try:
        return (2 * (TP * TN - FN * FP))/((TP + FP) * (FP + TN) + (TP + FN) * (FN + TN))
    except ZeroDivisionError:
        return np.inf


if __name__ == "__main__":
    FILE_NAME_NEG = "mouse_negative"
    FILE_NAME_POS = "mouse_positive"
    FILE_NAME_NEG_CTRL = "mouse_negative_with_ctrl"
    FILE_NAME_POS_CTRL = "mouse_positive_with_ctrl"

    USED_COLUMNS = ["a", "u", "d", "k1", "k2"]
    CLUSTERING_METHODS = [GaussianMixture(covariance_type="diag",
                                          n_components=2, n_init=10),
                          KMeans(n_clusters=2, n_init=10)]
    PERCENTAGES = [x/100 for x in range(101)]

    kappas = dict((c, []) for c in CLUSTERING_METHODS)
    for c in CLUSTERING_METHODS:
        for percentage in progress_bar(PERCENTAGES):
            kappas[c].append(np.mean([main(percentage, c) for _ in range(30)]))

    fig, ax = plt.subplots(1, 1)
    for c in CLUSTERING_METHODS:
        ax.plot(PERCENTAGES, kappas[c], label=c)
    ax.set_xlabel("percentage from positive control")
    ax.set_ylabel("cohen's kappa")
    ax.legend()
    plt.show()
