import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

    all_freqs, all_amps, all_phases = [], [], []

    ignore = np.loadtxt(f"intermediate/ignore_{file}.csv", delimiter=",").flatten()

    for particle_idx in all_parameters[:, indices.index("idx")]:
        if particle_idx in ignore:
            continue

        try:
            row_number = all_parameters[:, indices.index("idx")].tolist().index(particle_idx)
            freq = all_parameters[row_number][indices.index("freq")]
            amp = all_parameters[row_number][indices.index("amp")]
            phase = all_parameters[row_number][indices.index("phase")]

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
                single_particle_data['fit_sigmoid'] = sigmoid_and_linear_decreasing(single_particle_data['frame'], w1, w2, a, d, u, k1, k2)
                single_particle_data['residuum'] = single_particle_data['ratio'] - single_particle_data['fit_sigmoid']

                start = int((np.abs(single_particle_data['frame'] - t)).argmin())
                end = len(single_particle_data['frame'])

                single_particle_data['fit_sin'] = freq_to_func(freq, amp, phase,
                                                               start, np.array(single_particle_data["frame"][start:end]))

                single_particle_data['fit_total'] = single_particle_data['fit_sigmoid'] + single_particle_data['fit_sin']

                print(f"freq: {freq:.3f}, amp: {amp:.3}, phase: {phase:.3}")
                visualize(single_particle_data, titel=f"particle {int(particle_idx)}")

            all_freqs.append(freq)
            all_amps.append(amp)
            all_phases.append(phase)

        except RuntimeError as e:
            print(e)

    print(len(all_freqs))
    len_filtered = 0
    # filter according to amplitude
    for i in range(len(all_freqs)-1, -1, -1):
        if all_amps[i] <= 0.1 or all_freqs[i] < 0.005:
            all_freqs.pop(i)
            all_amps.pop(i)
            all_phases.pop(i)
            len_filtered += 1
    print(len_filtered)

    print(f"mean freq: {np.mean(all_freqs):.3f}, mean amp: {np.mean(all_amps):.3f}, mean phase: {np.mean(all_phases):.3f}")

    # transform phase to be within [-pi, pi]
    for i in range(len(all_phases)):
        if all_phases[i] > np.pi or all_phases[i] < -np.pi:
            all_phases[i] = all_phases[i] - int(all_phases[i] / np.pi) * 2 * np.pi

    # violin plots
    labels = []

    def add_label(violin, label):
        color = violin["bodies"][0].get_facecolor().flatten()
        labels.append((mpatches.Patch(color=color), label))

    fig, ax = plt.subplots(3)

    def set_plot_sizes(i):
        ax[i].xaxis.label.set_size(12)
        ax[i].yaxis.label.set_size(12)
        ax[i].tick_params(axis='both', labelsize=12)
        ax[i].legend(*zip(*labels), loc=1)

    labels = []
    add_label(ax[0].violinplot(all_freqs, showmedians=True, showmeans=False, showextrema=False), label="frequency")
    set_plot_sizes(0)

    labels = []
    add_label(ax[1].violinplot(np.abs(all_amps), showmedians=True, showmeans=False, showextrema=False), label="amplitude")
    set_plot_sizes(1)

    labels = []
    add_label(ax[2].violinplot(all_phases, showmedians=True, showmeans=False, showextrema=False), label="phase")
    set_plot_sizes(2)

    plt.show()

    # histogram
    fig, axs = plt.subplots(1)
    axs.set_title("weighted frequencies")
    axs.hist(all_freqs, weights=np.abs(all_amps), bins=20)
    plt.show()

    fig, axs = plt.subplots(1)
    axs.set_title("typical oscillation")
    axs.plot(np.arange(1000), freq_to_func(np.mean(all_freqs), np.mean(all_amps), np.mean(phase), 0, np.arange(1000)))
    plt.show()


if __name__ == '__main__':
    """
    statistical analysis of frequencies and amplitudes
    """

    main("human_positive")
    main("mouse_positive")
