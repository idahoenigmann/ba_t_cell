import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd


def import_all_data(files, equalize_weights=False):
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
        df2["activation"] = "positive" if file in ["human_positive", "mouse_positive"] else "negative"
        df2["cell_type"] = "human" if file in ["human_positive", "human_negative"] else "mouse"
        all_data.append(df2)
        weights.append(len(df2))

    if equalize_weights:
        for i in range(len(all_data)):
            all_data[i] = all_data[i].sample(min(weights))

    all_data = pd.concat(all_data)

    return all_data


def visualize_2d_compare(data, x_axis, y_axis):
    fig, ax = plt.subplots(1, 3, sharey=True)

    for i, filtered in [[0, "file"], [1, "predicted_clusters_gm"], [2, "predicted_clusters_kmeans"]]:
        for f in set(data[filtered]):
            data_f = data.loc[data[filtered] == f]
            ax[i].scatter(data_f[x_axis], data_f[y_axis], label=f, alpha=0.3)
        ax[i].set_xlabel(x_axis)
        ax[i].legend()
        ax[i].xaxis.label.set_size(12)
        ax[i].yaxis.label.set_size(12)
        ax[i].tick_params(axis='both', labelsize=12)
    ax[0].set_ylabel("PCA2")
    ax[0].set_title("Input Files")
    ax[1].set_title("Gaussian Mixture Model")
    ax[2].set_title("KMeans")
    plt.show()


def main(n_components, normalize, equalize_weights):
    data = import_all_data(["human_positive", "human_negative"], equalize_weights)

    if normalize:
        scaler = StandardScaler()
        data[["a", "u", "d", "k1", "k2", "w1", "w2"]] = scaler.fit_transform(data[["a", "u", "d", "k1", "k2", "w1", "w2"]])

    dim = 2
    prediction_parameters = ["a", "u", "d", "k1", "k2"]  # idx,start,s,w1,t,w2,e,a,d,u,k1,k2,mse_sigmoid,mse_total

    # clustering
    clustering_gm = GaussianMixture(n_components=n_components, covariance_type="diag", n_init=10)
    clustering_kmeans = KMeans(n_clusters=n_components, n_init=10)
    clustering_gm.fit(data[prediction_parameters])
    clustering_kmeans.fit(data[prediction_parameters])
    data["predicted_clusters_gm"] = clustering_gm.predict(data[prediction_parameters])
    data["predicted_clusters_kmeans"] = clustering_kmeans.predict(data[prediction_parameters])

    # do pca for visualization in 2d
    pca = PCA(n_components=2, whiten=True)
    pca.fit(data[prediction_parameters])

    new_data = pandas.DataFrame(pca.transform(data[prediction_parameters]),
                                columns=["PCA1", "PCA2"],
                                index=data[prediction_parameters].index)
    new_data["file"] = data["file"]
    new_data["predicted_clusters_gm"] = data["predicted_clusters_gm"]
    new_data["predicted_clusters_kmeans"] = data["predicted_clusters_kmeans"]

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

    main(2, True, True)
