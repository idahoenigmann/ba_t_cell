import matplotlib
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    matplotlib.use('TkAgg')

    all_parameters = np.loadtxt('particle_parameters.csv', delimiter=',')

    indices = ['idx', 'w', 't', 'e', 'a', 'd', 'u', 'k']

    fig, ax = plt.subplots(2)
    ax[0].set_title('u, a, d')
    ax[0].violinplot([param[indices.index('u')] for param in all_parameters])
    ax[0].violinplot([param[indices.index('a')] for param in all_parameters])
    ax[0].violinplot([param[indices.index('d')] for param in all_parameters])

    ax[1].set_title('w, t')
    ax[1].violinplot([param[indices.index('w')] for param in all_parameters], vert=False)
    ax[1].violinplot([param[indices.index('t')] for param in all_parameters], vert=False)
    plt.show()
