import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from venn import venn
from approximation import read_data, particle_to_parameters


def violin_plot_visualization(parameters: list, par_names: list, ax: matplotlib.pyplot.axes, vert: bool = True):
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

    ax.set_title(', '.join(par_names))
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.tick_params(axis='both', labelsize=12)
    for i in range(len(par_names)):
        add_label(ax.violinplot(parameters[i], showmedians=True, showmeans=False, showextrema=False, vert=vert), label=par_names[i])
    ax.legend(*zip(*labels), loc=1)


def plot_parameters(parameters: np.ndarray, par_indices: list):
    """
    shows violin plots of various parameters
    :param parameters: numpy matrix of all parameter values
    :param par_indices: list of names describing the column order of the parameters matrix
    """

    violin_plots = [['u', 'a', 'd'], ['w1', 'w2'], ['k1', 'k2']]
    # violin_plots = [['u', 'a', 'd'], ["start", 's', 'w1', 't', 'w2', 'e'], ['k1', 'k2']]

    fig, ax = plt.subplots(3)

    violin_plot_visualization([[param[par_indices.index(e)] for param in parameters] for e in violin_plots[0]],
                              violin_plots[0], ax[0], vert=True)
    violin_plot_visualization([[param[par_indices.index(e)] for param in parameters] for e in violin_plots[1]],
                              violin_plots[1], ax[1], vert=False)
    violin_plot_visualization([[param[par_indices.index(e)] for param in parameters] for e in violin_plots[2]],
                              violin_plots[2], ax[2], vert=True)

    plt.show()


def plot_error(parameters: np.ndarray, par_indices: list):
    """
    shows violin plots of errors
    :param parameters: numpy matrix of all parameter values
    :param par_indices: list of names describing the column order of the parameters matrix
    """

    violin_plot = ["mse_sigmoid", "mse_total"]
    fig, ax = plt.subplots(1)

    violin_plot_visualization([[param[par_indices.index(e)] for param in parameters] for e in violin_plot],
                              violin_plot, ax, vert=True)

    plt.show()


def statistics(values: list) -> dict:
    """
    calculate various statistics to values given
    :param values: list of numerical values
    :return: dictionary containing statistics
    """
    return {"min": np.min(values), "avg": np.average(values), "max": np.max(values), "std": np.std(values),
            "mean": np.mean(values)}


def find_outlier(values: np.ndarray, width: float, par_indices: list, par_used: list = []) -> dict:
    """
    returns sets of outliers, where an outlier is defined by being more than width*std_dev away from the average
    :param par_used: list of parameter names, to be used when finding outliers
    :param par_indices: list of names describing the column order of the parameters matrix
    :param values: matrix of values for which to determine the outliers
    :param width: min distance from mean in std for particle to count as outlier
    """

    outliers = dict()

    for par in set(par_used):
        stat = statistics(values[:, par_indices.index(par)].tolist())
        mean, std = stat["mean"], stat["std"]

        outliers[par] = set([values[i, par_indices.index("idx")] for i in range(values.shape[0])
                             if abs(values[i, par_indices.index(par)] - mean) > std * width])
    return outliers


def main(file):
    # matplotlib.use('TkAgg')

    # read parameters from file
    all_parameters = np.loadtxt(f'intermediate/particle_parameters_{file}.csv', delimiter=',')
    with open(f'intermediate/particle_parameters_{file}.csv', 'r') as f_in:
        header = f_in.readline()
        header = header.translate({ord(c): None for c in '# \n'})
        indices = header.split(',')

    # print statistics for all parameters
    print("Statistics for all parameters")
    for e in indices:
        print(f"{e}: {statistics(all_parameters[:, indices.index(e)].tolist())}")
    print(f"t-s: {statistics(all_parameters[:, indices.index('t')] - all_parameters[:, indices.index('s')].tolist())}")
    if "freq1" in indices:
        print(f"weighted average freq: {np.average([all_parameters[:, indices.index(f'freq{i}')] for i in range(10)], 
                                            weights=[all_parameters[:, indices.index(f'amp{i}')] for i in range(10)])}")
    print()

    # find outliers in parameters
    print("Outlier analysis")
    outliers = find_outlier(all_parameters, 2, indices, par_used=["a", "d"])
    print(f"Total number of particles: {all_parameters.shape[0]}")
    print(f"Total number of outliers: {len(set([item for sublist in outliers.values() for item in sublist]))}")

    for e in outliers.keys():
        stat = statistics(all_parameters[:, indices.index(e)].tolist())
        mean, std = stat["mean"], stat["std"]
        print(f"Outliers in {e}: {len(outliers[e])}            interval: [{mean - 2*std}, {mean + 2*std}]")
    print()

    # show distribution of parameters
    plot_parameters(all_parameters, indices)
    plot_error(all_parameters, indices)

    # show venn diagram of outliers
    if 2 <= len(outliers.keys()) <= 6:
        fig, ax = plt.subplots(1)
        ax.set_title("Outliers")
        venn(outliers, ax=ax)
        plt.show()

    # plot outliers
    data = read_data(file)
    for particle_idx in list(set([item for sublist in outliers.values() for item in sublist])):
        try:
            particle_to_parameters(data.loc[data['particle'] == particle_idx][['frame', 'ratio']],
                                   output_information=False, visualize_particles=True, select_by_input=False,
                                   titel=f"{int(particle_idx)} is outlier in parameters {",".join(
                                       [key for key in outliers.keys() if particle_idx in outliers[key]])}")
        except RuntimeError as e:
            print(e)


if __name__ == '__main__':
    """
    statistical analysis of parameters, plots and prints information
    """
    main("human_negative")
