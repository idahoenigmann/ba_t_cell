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


def import_all_data(files, normalize=True, equalize_weights=False):
    all_data = []
    weights = []
    for file in files:
        try:
            ignore = np.loadtxt(f"intermediate/ignore_{file}.csv")
        except FileNotFoundError:
            ignore = []

        with open(f'intermediate/particle_parameters_{file}.csv', 'r') as f_in:
            header = f_in.readline()
            header = header.translate({ord(c): None for c in '# \n'}).split(',')
        df2 = pd.DataFrame(data=np.loadtxt(f'intermediate/particle_parameters_{file}.csv', delimiter=','),
                           columns=header)
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

    if normalize:
        scaler = StandardScaler()
        all_data[["a", "u", "d", "k1", "k2", "w1", "w2"]] = (
            scaler.fit_transform(all_data[["a", "u", "d", "k1", "k2", "w1", "w2"]]))

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


def main(n_components, clustering_method, files, normalize, equalize_weights, vis_individual):
    print(f"{n_components} components on files {files} with {clustering_method}")

    data = import_all_data(files, normalize, equalize_weights)
    dim = 2
    prediction_parameters = ["a", "u", "d", "k1", "k2", "w1", "w2"]  # idx,start,s,w1,t,w2,e,a,d,u,k1,k2,mse_sigmoid,mse_total

    # clustering
    if clustering_method == "gaussian_mixture":
        clustering = GaussianMixture(n_components=n_components, covariance_type="diag", n_init=10)
    elif clustering_method == "kmeans":
        clustering = KMeans(n_clusters=n_components, n_init=10)
    data["predicted_clusters"] = clustering.fit_predict(whiten(data[prediction_parameters]))

    res = range(n_components)
    # find association between predicted clusters and files
    if n_components == len(files):
        assign_matrix = np.zeros([len(files), n_components])
        for i in range(n_components):
            for f_idx in range(len(files)):
                assign_matrix[f_idx, i] = len(data[(data['predicted_clusters'] == i) &
                                                   (data['file'] == files[f_idx])])
                print(f"{i} + {files[f_idx]}: {assign_matrix[f_idx, i]}")
            print()

        res_sum = 0
        for per in itertools.permutations(range(len(files))):
            tmp_sum = 0
            for i in range(len(per)):
                tmp_sum += assign_matrix[per[i], i]
            if tmp_sum > res_sum:
                res = per
                res_sum = tmp_sum

        # print out means and std
        for i in range(len(files)):
            print(files[res[i]])
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

    main(2, "gaussian_mixture", ["mouse_positive", "mouse_negative", "mouse_experiment"], True, False, True)
