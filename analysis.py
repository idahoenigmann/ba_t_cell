import matplotlib
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    matplotlib.use('TkAgg')

    all_parameters = np.loadtxt('particle_parameters.csv', delimiter=',')

    with open('particle_parameters.csv', 'r') as f_in:
        header = f_in.readline()
        header = header.translate({ord(c): None for c in '# \n'})
        indices = header.split(',')

    violin_plots = [['u', 'a', 'd'], ['w', 't']]

    fig, ax = plt.subplots(2)
    ax[0].set_title(','.join(violin_plots[0]))
    for e in violin_plots[0]:
        if e in indices:
            ax[0].violinplot([param[indices.index(e)] for param in all_parameters])

    ax[1].set_title(','.join(violin_plots[1]))
    for e in violin_plots[1]:
        if e in indices:
            ax[1].violinplot([param[indices.index(e)] for param in all_parameters], vert=False)

    plt.show()
