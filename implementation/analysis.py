import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from venn import venn
from approximation import read_data, visualize, sigmoid_and_linear_decreasing
from matplotlib.backends.backend_pdf import PdfPages


def violin_plot_visualization(parameters: pd.DataFrame, ax: matplotlib.pyplot.axes, vert: bool = True):
    """
    plots violin plots of various parameters
    :param vert: orientation of violin plot
    :param ax: axis on which to plot
    :param parameters: numpy matrix of all parameter values
    :param par_names: list of names describing the parameters matrix
    """
    labels = []

    def add_label(violin, label):
        color = violin["bodies"][0].get_facecolor().flatten()
        labels.append((mpatches.Patch(color=color), label))

    ax.set_title(', '.join(parameters.columns))
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.tick_params(axis='both', labelsize=12)
    for e in parameters.columns:
        add_label(ax.violinplot(parameters[e], showmedians=True, showmeans=False, showextrema=False, vert=vert),
                  label=e)
    ax.legend(*zip(*labels), loc=1)


def plot_parameters(parameters: pd.DataFrame):
    """
    shows violin plots of various parameters
    :param parameters: numpy matrix of all parameter values
    """

    violin_plots = [['u', 'a', 'd'], ['w1', 'w2'], ['k1', 'k2']]
    # violin_plots = [['u', 'a', 'd'], ["start", 's', 'w1', 't', 'w2', 'e'], ['k1', 'k2']]

    fig, ax = plt.subplots(3)

    violin_plot_visualization(parameters[violin_plots[0]], ax[0], vert=True)
    violin_plot_visualization(parameters[violin_plots[1]], ax[1], vert=False)
    violin_plot_visualization(parameters[violin_plots[2]], ax[2], vert=True)

    plt.show()


def plot_error(parameters: pd.DataFrame):
    """
    shows violin plots of errors
    :param parameters: numpy matrix of all parameter values
    """

    violin_plot = ["mse_sigmoid", "mse_total"]
    fig, ax = plt.subplots(1)

    violin_plot_visualization(parameters[violin_plot], ax, vert=True)

    plt.show()


def statistics(values: list) -> dict:
    """
    calculate various statistics to values given
    :param values: list of numerical values
    :return: dictionary containing statistics
    """
    return {"min": np.min(values), "avg": np.average(values), "max": np.max(values), "std": np.std(values),
            "mean": np.mean(values)}


def find_outlier(values: pd.DataFrame, width: list, par_used: list = ()) -> dict:
    """
    returns sets of outliers, where an outlier is defined by being more than width*std_dev away from the average
    :param par_used: list of parameter names, to be used when finding outliers
    :param values: matrix of values for which to determine the outliers
    :param width: min distance from mean in std for particle to count as outlier
    """

    outliers = dict()

    for par in set(par_used):
        stat = statistics(values[par].tolist())
        mean, std = stat["mean"], stat["std"]

        interval = [mean - std * width[0], mean + std * width[1]]

        outliers[par] = set([values["idx"][i] for i in range(values.shape[0])
                             if values[par][i] < interval[0] or
                             values[par][i] > interval[1]])
    return outliers


def main(file, width, par_used, ignore_file=True):
    # matplotlib.use('TkAgg')

    SAVE_PDF = False
    SHOW = False

    # read parameters from file
    all_parameters = pd.DataFrame(pd.read_hdf(f"intermediate/par_{file}.h5", "parameters"))

    # print statistics for all parameters
    print("Statistics for all parameters")
    for e in all_parameters.columns:
        print(f"{e}: {statistics(all_parameters[e].tolist())}")
    print()

    if SAVE_PDF:
        pp = PdfPages(f'{file}.pdf')

    # find outliers in parameters
    outliers = find_outlier(all_parameters, width, par_used=par_used)

    text = (f"\n{file}\n\nOutlier analysis\n"
            f"Total number of particles: {all_parameters.shape[0]}\n"
            f"Total number of outliers: {len(set([item for sublist in outliers.values() for item in sublist]))}\n\n")

    for e in outliers.keys():
        stat = statistics(all_parameters[e].tolist())
        mean, std = stat["mean"], stat["std"]
        text += (f"Outliers in {e}: {len(outliers[e])}\n"
                 f"mean: {mean}\n"
                 f"std: {std}\n"
                 f"interval: [{mean - width[0] * std}, {mean + width[1] * std}]\n\n")
    print(text)

    if SAVE_PDF:
        fig = plt.figure()
        plt.axis((0, 10, 0, 10))
        fig.axes[0].get_xaxis().set_visible(False)
        fig.axes[0].get_yaxis().set_visible(False)
        plt.text(5, 10, text, fontsize=18, style='oblique', ha='center',
                 va='top', wrap=True)
        pp.savefig(fig)
        plt.close(fig)

    # show distribution of parameters
    plot_parameters(all_parameters)
    plot_error(all_parameters)

    # show venn diagram of outliers
    if 2 <= len(outliers.keys()) <= 6:
        fig, ax = plt.subplots(1)
        ax.set_title("Outliers")
        venn(outliers, ax=ax)
        plt.show()

    # plot outliers
    data = read_data(file)

    for particle_idx in list(set([item for sublist in outliers.values() for item in sublist])):
        outlier_params = [key for key, val in outliers.items() if particle_idx in val]

        try:
            single_data = all_parameters[all_parameters["idx"] == particle_idx]
            w1 = single_data["w1"].tolist()[0]
            w2 = single_data["w2"].tolist()[0]
            a = single_data["a"].tolist()[0]
            d = single_data["d"].tolist()[0]
            u = single_data["u"].tolist()[0]
            k1 = single_data["k1"].tolist()[0]
            k2 = single_data["k2"].tolist()[0]

            single_particle_data = data.loc[data['particle'] == particle_idx][['frame', 'ratio']]
            single_particle_data['fit_sigmoid'] = sigmoid_and_linear_decreasing(single_particle_data['frame'], w1, w2,
                                                                                a, d, u, k1, k2)
            fig = visualize(single_particle_data, titel=f"particle {int(particle_idx)}: {str(outlier_params)}", return_fig=True)
            if SAVE_PDF:
                pp.savefig(fig)
            if SHOW:
                plt.show()
            plt.close(fig)

        except RuntimeError as e:
            print(e)

    if SAVE_PDF:
        pp.close()

    if ignore_file:
        ignore = [list(out)[e] for out in outliers.values() for e in range(len(out))]
        np.savetxt(f"intermediate/ignore_{file}.csv", list(set(ignore)))


if __name__ == '__main__':
    """
    statistical analysis of parameters, plots and prints information
    """

    # filter out all outliers
    main("human_positive", [3, 3], ["a", "u", "d", "k1", "k2"], ignore_file=True)


    # filter out pre-activated cells
    # main("human_positive", [np.infty, 0.5], ["u"], ignore_file=False)
    # main("mouse_positive", [np.infty, 1], ["u"], ignore_file=False)
