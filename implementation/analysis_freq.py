import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas

from approximation import read_data, visualize, sigmoid_and_linear_decreasing, freq_to_func


def main(file):
    # matplotlib.use('TkAgg')

    VISUALIZE = False

    print(file)

    # read parameters from file
    all_parameters = np.loadtxt(f'intermediate/particle_parameters_{file}.csv', delimiter=',')
    with open(f'intermediate/particle_parameters_{file}.csv', 'r') as f_in:
        header = f_in.readline()
        header = header.translate({ord(c): None for c in '# \n'})
        indices = header.split(',')

    data = read_data(file)

    all_freqs = []
    all_amps = []

    ignore = np.loadtxt(f"intermediate/ignore_{file}.csv", delimiter=",").flatten()

    for particle_idx in all_parameters[:, indices.index("idx")]:
        if particle_idx in ignore:
            continue

        try:
            row_number = all_parameters[:, indices.index("idx")].tolist().index(particle_idx)
            freqs = [all_parameters[row_number][indices.index(f"freq{i}")] for i in range(10)]
            amps = [all_parameters[row_number][indices.index(f"amp{i}")] for i in range(10)]

            tmp = list(zip(*((f, a) for f, a in zip(freqs, amps) if a > 0)))
            if len(tmp) == 2:
                freqs, amps = tmp
            else:
                freqs, amps = [], []

            freqs, amps = np.array(freqs, dtype=int), np.array(amps)

            if VISUALIZE:
                w1 = all_parameters[row_number][indices.index("w1")]
                w2 = all_parameters[row_number][indices.index("w2")]
                a = all_parameters[row_number][indices.index("a")]
                d = all_parameters[row_number][indices.index("d")]
                u = all_parameters[row_number][indices.index("u")]
                k1 = all_parameters[row_number][indices.index("k1")]
                k2 = all_parameters[row_number][indices.index("k2")]
                t = all_parameters[row_number][indices.index("t")]

                single_particle_data = data.loc[data['particle'] == particle_idx][['frame', 'ratio']]
                single_particle_data['fit_sigmoid'] = sigmoid_and_linear_decreasing(single_particle_data['frame'], w1, w2,
                                                                                    a, d, u, k1, k2)
                single_particle_data['residuum'] = single_particle_data['ratio'] - single_particle_data['fit_sigmoid']

                single_particle_data['fit_sin'] = freq_to_func(freqs, amps,
                                                               int((np.abs(single_particle_data['frame'] - t)).argmin()),
                                                               len(single_particle_data['frame']))

                single_particle_data['fit_total'] = single_particle_data['fit_sigmoid'] + single_particle_data['fit_sin']

                visualize(single_particle_data, titel=f"particle {int(particle_idx)}")

            all_freqs.extend(freqs)
            all_amps.extend(amps)

        except RuntimeError as e:
            print(e)

    # filter according to amplitude
    try:
        all_freqs, all_amps = zip(*((f, a) for f, a in zip(all_freqs, all_amps) if 2 <= f <= 20 and a > 0))
    except ValueError:
        print("no frequencies and amplitudes matched the given conditions")
        return

    # violin plots
    labels = []

    def add_label(violin, label):
        color = violin["bodies"][0].get_facecolor().flatten()
        labels.append((mpatches.Patch(color=color), label))

    fig, ax = plt.subplots(2)
    ax[0].set_title("freqs")
    ax[0].xaxis.label.set_size(12)
    ax[0].yaxis.label.set_size(12)
    ax[0].tick_params(axis='both', labelsize=12)
    add_label(ax[0].violinplot(all_freqs, showmedians=True, showmeans=False, showextrema=False),
              label="freqs")
    ax[0].legend(*zip(*labels), loc=1)

    labels = []
    ax[1].set_title("amps")
    ax[1].xaxis.label.set_size(12)
    ax[1].yaxis.label.set_size(12)
    ax[1].tick_params(axis='both', labelsize=12)
    add_label(ax[1].violinplot(all_amps, showmedians=True, showmeans=False, showextrema=False),
              label="amps")
    ax[1].legend(*zip(*labels), loc=1)
    plt.show()

    tmp_freqs = np.array(list(set(all_freqs)), dtype=int)
    tmp_amps = np.array([sum([all_amps[i] for i in range(len(all_amps)) if all_freqs[i] == f]) for f in tmp_freqs])

    # histogram
    fig, axs = plt.subplots(1)
    axs.set_title("weighted frequencies")
    axs.hist(tmp_freqs, weights=tmp_amps, bins=20)
    plt.show()

    # typical oscillation
    tmp = pandas.DataFrame()
    tmp["frame"] = np.array(range(0, 1000))
    tmp["ratio"] = freq_to_func(tmp_freqs, tmp_amps, 0, 1000)
    visualize(tmp)


if __name__ == '__main__':
    """
    statistical analysis of frequencies and amplitudes
    """

    """
    # Test different frequenices and amplitudes
    tmp = pandas.DataFrame()
    tmp["frame"] = np.array(range(0, 1000))
    tmp["ratio"] = freq_to_func(np.array([10], dtype=int), np.array([500]), 0, 1000)
    visualize(tmp)
    """

    # main("human_positive")
    # main("human_negative")
    # main("mouse_positive")
    main("mouse_negative")
