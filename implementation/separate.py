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


def index_term_to_data(data, term):
    def term_sub_helper(data, term):
        token_lst = term.split("-")
        if len(token_lst) == 1:
            return data[term]
        else:
            tmp = data[token_lst[0]]
            for i in range(1, len(token_lst)):
                tmp -= data[token_lst[i]]
            return tmp

    token_lst = term.split("+")
    if len(token_lst) == 1:
        return term_sub_helper(data, term)
    else:
        tmp = term_sub_helper(data, token_lst[0])
        for i in range(1, len(token_lst)):
            tmp += term_sub_helper(data, token_lst[i])
        return tmp


def visualize_2d_compare(data, x_axis, y_axis):
    fig, ax = plt.subplots(1, 2)

    for i, filtered in [[0, "file"], [1, "predicted_clusters"]]:
        for f in set(data[filtered]):
            data_f = data.loc[data[filtered] == f]
            ax[i].scatter(index_term_to_data(data_f, x_axis), index_term_to_data(data_f, y_axis), label=f, alpha=0.3)
        ax[i].set_xlabel(x_axis)
        ax[i].set_ylabel(y_axis)
        ax[i].legend()
        ax[i].xaxis.label.set_size(12)
        ax[i].yaxis.label.set_size(12)
        ax[i].tick_params(axis='both', labelsize=12)
    plt.show()


def visualize_3d_compare(data, x_axis, y_axis, z_axis):
    for i, filtered in [[0, "file"], [1, "predicted_clusters"]]:
        fig = plt.figure(i)
        ax = fig.add_subplot(projection='3d')

        for f in set(data[filtered]):
            data_f = data.loc[data[filtered] == f]
            ax.scatter(index_term_to_data(data_f, x_axis), index_term_to_data(data_f, y_axis),
                          index_term_to_data(data_f, z_axis), label=f)
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.set_zlabel(z_axis)
        ax.legend()
    plt.show()


def main(n_components, clustering_method, normalize, equalize_weights, vis_individual):
    print(f"{n_components} components with {clustering_method}")

    fit_files = ["mouse_positive", "mouse_negative"]
    ctrl_files = [] # ["mouse_positive_with_ctrl", "mouse_negative_with_ctrl"]
    files = ["mouse_experiment", "mouse_positive", "mouse_negative"] # ["mouse_positive_with_ctrl", "mouse_negative_with_ctrl", "mouse_positive", "mouse_negative"]

    print(f"fit with {fit_files}, show with {files}")

    data_no_ctrl = import_all_data(fit_files, equalize_weights)
    data = import_all_data(files, False)

    if normalize:
        scaler = StandardScaler()
        data_no_ctrl[["a", "u", "d", "k1", "k2", "w1", "w2"]] = scaler.fit_transform(data_no_ctrl[["a", "u", "d", "k1", "k2", "w1", "w2"]])
        data[["a", "u", "d", "k1", "k2", "w1", "w2"]] = scaler.transform(data[["a", "u", "d", "k1", "k2", "w1", "w2"]])

    dim = 2
    # NOTE: It probably makes sense to remove w1 and w2 and maybe add w2-w1 instead
    prediction_parameters = ["a", "u", "d", "k1", "k2", "w1", "w2"]  # idx,start,s,w1,t,w2,e,a,d,u,k1,k2,mse_sigmoid,mse_total

    # clustering
    if clustering_method == "gaussian_mixture":
        clustering = GaussianMixture(n_components=n_components, covariance_type="diag", n_init=10)
    elif clustering_method == "kmeans":
        clustering = KMeans(n_clusters=n_components, n_init=10)
    clustering.fit(data_no_ctrl[prediction_parameters])
    data["predicted_clusters"] = clustering.predict(data[prediction_parameters])

    res = range(n_components)
    # find association between predicted clusters and files
    if n_components == len(ctrl_files):
        assign_matrix = np.zeros([len(ctrl_files), n_components])
        for i in range(n_components):
            for f_idx in range(len(ctrl_files)):
                assign_matrix[f_idx, i] = len(data[(data['predicted_clusters'] == i) &
                                                   (data['file'] == ctrl_files[f_idx])])
                print(f"{i} + {ctrl_files[f_idx]}: {assign_matrix[f_idx, i]}")
            print()

        res_sum = 0
        for per in itertools.permutations(range(len(ctrl_files))):
            tmp_sum = 0
            for i in range(len(per)):
                tmp_sum += assign_matrix[per[i], i]
            if tmp_sum > res_sum:
                res = per
                res_sum = tmp_sum

        # print out means and std
        for i in range(len(ctrl_files)):
            print(ctrl_files[res[i]])
            if clustering_method == "gaussian_mixture":
                print(f"weigh: {clustering.weights_[i]}")
                print(f"mean: {clustering.means_[i]}")
                print(f"covariance: {clustering.covariances_[i]}")
            elif clustering_method == "kmeans":
                print(f"centre: {clustering.cluster_centers_[i]}")
            print()

    # do pca for visualization in 2d
    pca = PCA(n_components=2, whiten=True)
    pca.fit(data[prediction_parameters])

    new_data = pandas.DataFrame(pca.transform(data[prediction_parameters]),
                                columns=["PCA1", "PCA2"],
                                index=data[prediction_parameters].index)
    new_data["file"] = data["file"]
    new_data["predicted_clusters"] = data["predicted_clusters"]

    visualize_2d_compare(new_data, "PCA1", "PCA2")

    if vis_individual:
        # visualization in all parameters
        if dim == 2:
            for x in range(len(prediction_parameters)):
                for y in range(x + 1, len(prediction_parameters)):
                    visualize_2d_compare(data, prediction_parameters[x], prediction_parameters[y])
        elif dim == 3:
            for x in range(len(prediction_parameters)):
                for y in range(x + 1, len(prediction_parameters)):
                    for z in range(y + 1, len(prediction_parameters)):
                        visualize_3d_compare(data, prediction_parameters[x],
                                             prediction_parameters[y], prediction_parameters[z])


if __name__ == "__main__":
    """
    plot multiple datasets in single plot with various axis,
    cluster the data
    """

    main(2, "gaussian_mixture", True, True, True)
