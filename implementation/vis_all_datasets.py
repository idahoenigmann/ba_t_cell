import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.cluster.vq import whiten
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import itertools


def import_all_data(files):
    all_data = []
    weights = []
    for file in files:
        try:
            ignore = np.loadtxt(f"intermediate/ignore_{file}.csv")
        except FileNotFoundError:
            ignore = []

        df2 = pd.DataFrame(pd.read_hdf(f"intermediate/par_{file}.h5", "parameters"))
        df2 = df2.drop(df2[df2["idx"].isin(ignore)].index)
        df2["file"] = file
        all_data.append(df2)
        weights.append(len(df2))

    return pd.concat(all_data)


def visualize_2d_compare(data, x_axis, y_axis):
    fig, ax = plt.subplots(1, 1)

    for f in set(data["file"]):
        data_f = data.loc[data["file"] == f]
        ax.scatter(data_f[x_axis], data_f[y_axis], label=f, alpha=0.3)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.legend()
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.tick_params(axis='both', labelsize=12)
    plt.show()


def main():
    data = import_all_data(["human_positive", "human_negative", "mouse_positive", "mouse_negative"])

    prediction_parameters = ["a", "u", "d", "k1", "k2"]

    # do pca for visualization in 2d
    pca = PCA(n_components=2, whiten=True)
    pca.fit(data[prediction_parameters])

    new_data = pandas.DataFrame(pca.transform(data[prediction_parameters]),
                                columns=["PCA1", "PCA2"],
                                index=data[prediction_parameters].index)
    new_data["file"] = data["file"]

    visualize_2d_compare(new_data, "PCA1", "PCA2")

    # visualization in all parameters
    for x in range(len(prediction_parameters)):
        for y in range(x + 1, len(prediction_parameters)):
            visualize_2d_compare(data, prediction_parameters[x], prediction_parameters[y])


if __name__ == "__main__":
    """
    plot multiple datasets in single plot with various axis,
    cluster the data
    """

    main()
