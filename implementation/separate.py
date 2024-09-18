import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import pandas as pd


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

    data = import_all_data()
    dim = 2
    tmp = ["a", "u", "d", "k1", "k2", "w1", "w2"]   # idx,start,s,w1,t,w2,e,a,d,u,k1,k2,mse_sigmoid,mse_total

    gm = GaussianMixture(n_components=2, covariance_type="diag")
    gm.fit(data[tmp])

    for i in range(dim):
        print(f"weigh: {gm.weights_[0]}")
        print(f"mean: {gm.means_[0]}")
        print(f"covariance: {gm.covariances_[0]}")
        print()

    data["predicted_clusters"] = gm.predict(data[tmp])
    print(f"0 + positive: {len(data[(data['predicted_clusters'] == 0) & (data['activation'] == 'positive')])}")
    print(f"1 + positive: {len(data[(data['predicted_clusters'] == 1) & (data['activation'] == 'positive')])}")
    print(f"0 + negative: {len(data[(data['predicted_clusters'] == 0) & (data['activation'] == 'negative')])}")
    print(f"1 + negative: {len(data[(data['predicted_clusters'] == 1) & (data['activation'] == 'negative')])}")

    if dim == 2:
        for x in range(len(tmp)):
            for y in range(x + 1, len(tmp)):
                visualize_2d_compare(data, tmp[x], tmp[y])
    elif dim == 3:
        for x in range(len(tmp)):
            for y in range(x + 1, len(tmp)):
                for z in range(y + 1, len(tmp)):
                    visualize_3d_compare(data, tmp[x], tmp[y], tmp[z])
