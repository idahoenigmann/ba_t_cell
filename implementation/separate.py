import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import pandas as pd
import itertools


def import_all_data():
    all_data = []
    for file in ["human_positive", "human_negative", "mouse_positive", "mouse_negative"]:
        with open(f'intermediate/particle_parameters_{file}.csv', 'r') as f_in:
            header = f_in.readline()
            header = header.translate({ord(c): None for c in '# \n'}).split(',')
        df2 = pd.DataFrame(data=np.loadtxt(f'intermediate/particle_parameters_{file}.csv', delimiter=','),
                           columns=header)
        df2["file"] = file
        df2["activation"] = "positive" if file in ["human_positive", "mouse_positive"] else "negative"
        df2["cell_type"] = "human" if file in ["human_positive", "human_negative"] else "mouse"
        all_data.append(df2)
    return pd.concat(all_data)


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
    fig, ax = plt.subplots(2)

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


if __name__ == "__main__":
    """
    plot multiple datasets in single plot with various axis,
    cluster the data
    """

    N_COMPONENTS = 4

    data = import_all_data()
    dim = 2
    tmp = ["a", "u", "d", "k1", "k2", "w1", "w2"]   # idx,start,s,w1,t,w2,e,a,d,u,k1,k2,mse_sigmoid,mse_total

    gm = GaussianMixture(n_components=N_COMPONENTS, covariance_type="diag")
    gm.fit(data[tmp])

    data["predicted_clusters"] = gm.predict(data[tmp])

    files = ["human_positive", "human_negative", "mouse_positive", "mouse_negative"]
    assign_matrix = np.zeros([len(files), N_COMPONENTS])
    for i in range(N_COMPONENTS):
        for f_idx in range(len(files)):
            assign_matrix[f_idx, i] = len(data[(data['predicted_clusters'] == i) & (data['file'] == files[f_idx])])
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

    for i in range(N_COMPONENTS):
        print(files[res[i]])
        print(f"weigh: {gm.weights_[i]}")
        print(f"mean: {gm.means_[i]}")
        print(f"covariance: {gm.covariances_[i]}")
        print()

    if dim == 2:
        for x in range(len(tmp)):
            for y in range(x + 1, len(tmp)):
                visualize_2d_compare(data, tmp[x], tmp[y])
    elif dim == 3:
        for x in range(len(tmp)):
            for y in range(x + 1, len(tmp)):
                for z in range(y + 1, len(tmp)):
                    visualize_3d_compare(data, tmp[x], tmp[y], tmp[z])
