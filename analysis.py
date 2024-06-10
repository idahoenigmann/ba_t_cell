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
    for i in range(len(par_names)):
        add_label(ax.violinplot(parameters[i], showmeans=True, showextrema=False, vert=vert), label=par_names[i])
    ax.legend(*zip(*labels), loc=1)


def plot_parameters(parameters: np.ndarray, par_indices: list):
    """
    shows violin plots of various parameters
    :param parameters: numpy matrix of all parameter values
    :param par_indices: list of names describing the column order of the parameters matrix
    """

    violin_plots = [['u', 'a', 'd'], ['s', 'w', 't', 'e'], ['k']]
    fig, ax = plt.subplots(3)

    violin_plot_visualization([[param[par_indices.index(e)] for param in parameters] for e in violin_plots[0]],
                              violin_plots[0], ax[0], vert=True)
    violin_plot_visualization([[param[par_indices.index(e)] for param in parameters] for e in violin_plots[1]],
                              violin_plots[1], ax[1], vert=False)
    violin_plot_visualization([[param[par_indices.index(e)] for param in parameters] for e in violin_plots[1]],
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


def find_outlier(values: np.ndarray, width: float, par_indices: list, ignore: list = None) -> dict:
    """
    returns sets of outliers, where an outlier is defined by being more than width*std_dev away from the average
    :param ignore: list of parameter names, to be ignored when finding outliers
    :param par_indices: list of names describing the column order of the parameters matrix
    :param values: matrix of values for which to determine the outliers
    :param width: list of names describing the column order of the values matrix
    """

    if ignore is None:
        ignore = []

    outliers = dict()

    for par in set(par_indices).difference(set(ignore)):
        stat = statistics(values[:, par_indices.index(par)].tolist())
        avg, std = stat["avg"], stat["std"]

        outliers[par] = set([values[i, par_indices.index("idx")] for i in range(values.shape[0])
                             if abs(values[i, par_indices.index(par)] - std) > std * width])
    return outliers


def main():
    # matplotlib.use('TkAgg')

    all_parameters = np.loadtxt('particle_parameters.csv', delimiter=',')

    with open('particle_parameters.csv', 'r') as f_in:
        header = f_in.readline()
        header = header.translate({ord(c): None for c in '# \n'})
        indices = header.split(',')

    # print statistics for all parameters
    print("Statistics for all parameters")
    for e in indices:
        print(f"{e}: {statistics(all_parameters[:, indices.index(e)].tolist())}")
    print(f"t-s: {statistics(all_parameters[:, indices.index('t')] - all_parameters[:, indices.index('s')].tolist())}")
    print()

    # find outliers in parameters
    print("Outlier analysis")
    outliers = find_outlier(all_parameters, 2.5, indices, ignore=["idx", "mse_sigmoid", "mse_total", 'e', 's', 't'])
    print(f"Total number of particles: {all_parameters.shape[0]}")

    for e in outliers.keys():
        print(f"Outliers in {e}: {len(outliers[e])}")
    print()

    # find specific outliers
    specify_par = ['k']
    outlier_particles = set(all_parameters[:, indices.index("idx")])
    for e in outliers.keys():
        if e in specify_par:
            outlier_particles = outlier_particles.intersection(outliers[e])
        else:
            outlier_particles = outlier_particles.difference(outliers[e])

    print(f"(Just) in {specify_par} are the following particles: {outlier_particles}")

    # show distribution of parameters
    plot_parameters(all_parameters, indices)
    plot_error(all_parameters, indices)

    # show venn diagram of outliers
    fig, ax = plt.subplots(1)
    ax.set_title("Outliers")
    venn(outliers, ax=ax)
    plt.show()

    # plot specific outliers
    data = read_data()
    for particle_idx in outlier_particles:
        particle_to_parameters(data.loc[data['particle'] == particle_idx][['frame', 'ratio']], output_information=True,
                               visualize_particles=True, select_by_input=False)


if __name__ == '__main__':
    """
    statistical analysis of parameters, plots and prints information
    """
    main()
