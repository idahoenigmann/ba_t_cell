import numpy as np
import matplotlib.pyplot as plt
from approximation import read_data


def main(file_name):
    # matplotlib.use('TkAgg')

    data = read_data(file_name)

    # filter out nan and inf values as well as too low and high values
    data = data[np.isfinite(data["ratio"])]
    data = data[np.less(data["ratio"], np.full((len(data["ratio"])), 5))]
    data = data[np.greater(data["ratio"], np.full((len(data["ratio"])), 0))]

    fig, ax = plt.subplots(1, figsize=(4, 4), dpi=140)
    ax.set_ylim(0, 5)
    ax.set_xlim(0, 1000)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.set_ylabel('ratio')
    ax.tick_params(axis='both', labelsize=12)

    for particle_idx in set(data['particle']):
        # get data of a single particle
        single_particle_data = data.loc[data['particle'] == particle_idx][['frame', 'ratio']]

        # skip if too few datapoints
        if len(single_particle_data['frame']) < 300:
            continue

        single_particle_data.plot(x="frame", y="ratio", color="#000000", alpha=0.015, ax=ax, legend=False)

    plt.savefig("visualizations/all_cells_overlayed.jpg")
    plt.show()


if __name__ == '__main__':
    """
    Generate single plot of all particle concentrations.
    """

    # main("human_positive")
    main("mouse_positive")
    # main("mouse_negative")
