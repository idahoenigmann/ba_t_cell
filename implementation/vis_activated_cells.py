import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd


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


def visualize_2d(data, x_axis, y_axis):
    fig, ax = plt.subplots(1, 1)

    for f in set(data["file"]):
        data_f = data.loc[data["file"] == f]
        ax.scatter(data_f[x_axis], data_f[y_axis], label=f, alpha=0.3)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.legend(prop={'size': 15})
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)
    ax.tick_params(axis='both', labelsize=15)
    fig.tight_layout()
    plt.show()


def main(files, normalize, equalize_weights, vis_individual):
    data = import_all_data(files, normalize, equalize_weights)
    prediction_parameters = ["a", "u", "d", "k1", "k2", "w1", "w2"]  # idx,start,s,w1,t,w2,e,a,d,u,k1,k2,mse_sigmoid,mse_total

    # do pca for visualization in 2d
    pca = PCA(n_components=2, whiten=True)
    pca.fit(data[prediction_parameters])

    new_data = pandas.DataFrame(pca.transform(data[prediction_parameters]),
                                columns=["PCA1", "PCA2"],
                                index=data[prediction_parameters].index)
    new_data["file"] = data["file"]

    visualize_2d(new_data, "PCA1", "PCA2")

    if vis_individual:
        # visualization in all parameters
        for x in range(len(prediction_parameters)):
            for y in range(x + 1, len(prediction_parameters)):
                visualize_2d(data, prediction_parameters[x], prediction_parameters[y])


if __name__ == "__main__":
    """
    plot multiple datasets in single plot with various axis,
    cluster the data
    """

    main(["mouse_positive", "human_positive", "mouse_negative", "human_negative"], True, False, True)
