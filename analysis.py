import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def violin_plot_visualization(parameters, par_indices):
    """
    plots violin plots of various parameters
    :param parameters: numpy matrix of all parameter values
    :param par_indices: list of names describing the column order of the parameters matrix
    """
    violin_plots = [['u', 'a', 'd'], ['s', 'w', 't', 'e']]
    labels = []

    def add_label(violin, label):
        color = violin["bodies"][0].get_facecolor().flatten()
        labels.append((mpatches.Patch(color=color), label))

    fig, ax = plt.subplots(2)
    ax[0].set_title(','.join(violin_plots[0]))
    for e in violin_plots[0]:
        if e in par_indices:
            add_label(ax[0].violinplot([param[par_indices.index(e)] for param in parameters], showmeans=True,
                                       showextrema=False), label=e)
    ax[0].legend(*zip(*labels), loc=1)

    labels = []

    ax[1].set_title(','.join(violin_plots[1]))
    for e in violin_plots[1]:
        if e in par_indices:
            add_label(ax[1].violinplot([param[par_indices.index(e)] for param in parameters], vert=False,
                                       showmeans=True, showextrema=False), label=e)
    ax[1].legend(*zip(*labels), loc=1)

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

    violin_plot_visualization(all_parameters, indices)


if __name__ == '__main__':
    """
    statistical analysis of parameters, plots and prints information
    """
    main()
