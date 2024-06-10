import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def violin_plot_visualization(parameters, par_names, ax, vert=True):
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
        add_label(ax.violinplot(parameters[i], showmeans=True,
                                   showextrema=False, vert=vert), label=par_names[i])
    ax.legend(*zip(*labels), loc=1)


def plot_parameters(parameters, par_indices):
    """
    shows violin plots of various parameters
    :param parameters: numpy matrix of all parameter values
    :param par_indices: list of names describing the column order of the parameters matrix
    """

    violin_plots = [['u', 'a', 'd'], ['s', 'w', 't', 'e']]
    fig, ax = plt.subplots(2)

    violin_plot_visualization([[param[par_indices.index(e)] for param in parameters] for e in violin_plots[0]],
                              violin_plots[0], ax[0], vert=True)
    violin_plot_visualization([[param[par_indices.index(e)] for param in parameters] for e in violin_plots[1]],
                              violin_plots[1], ax[1], vert=False)

    plt.show()


def plot_error(parameters, par_indices):
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


def statistics(values):
    """
    calculate various statistics to values given
    :param values: list of numerical values
    :return: dictionary containing statistics
    """
    return {"min": np.min(values), "avg": np.average(values), "max": np.max(values), "std": np.std(values),
            "mean": np.mean(values)}


def main():
    matplotlib.use('TkAgg')

    all_parameters = np.loadtxt('particle_parameters.csv', delimiter=',')

    with open('particle_parameters.csv', 'r') as f_in:
        header = f_in.readline()
        header = header.translate({ord(c): None for c in '# \n'})
        indices = header.split(',')

    for e in indices:
        print(f"{e}: {statistics(all_parameters[:, indices.index(e)])}")
    print(f"t-s: {statistics(all_parameters[:, indices.index('t')] - all_parameters[:, indices.index('s')])}")

    plot_parameters(all_parameters, indices)
    plot_error(all_parameters, indices)


if __name__ == '__main__':
    """
    statistical analysis of parameters, plots and prints information
    """
    main()
