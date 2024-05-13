import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import math
import numpy as np
import scipy


def read_data():
    data = pd.read_hdf('data/SLB7_231218.h5')
    print(f"header: {[e for e in data.columns]}")

    # check which lengths of data are present
    # lengths = [len(data.loc[data['particle'] == idx][['frame']]) for idx in set(data['particle'])]
    # print(len([lengths[i] for i in range(len(lengths)) if lengths[i] == 948]))
    return data


def visualize_single_particle(single_particle_data):
    ax = single_particle_data.plot.scatter(x="frame", y="ratio")
    if 'fitted' in single_particle_data.columns:
        if 'fft' in single_particle_data.columns:
            single_particle_data['total_fit'] = single_particle_data['fitted'] + single_particle_data['fft']
            single_particle_data.plot(x='frame', y='total_fit', color="red", ax=ax)
        else:
            single_particle_data.plot(x='frame', y='fitted', color="red", ax=ax)
    plt.show()


def fit_basic_function(xdata, ydata):
    min_val, median_val, max_val = np.min(ydata), np.median(ydata), np.max(ydata)
    lower_bounds = (min_val, xdata[0], 0.05, median_val, min_val)
    upper_bounds = (median_val, xdata[-1], 10, max_val, max_val)
    p0 = (ydata[0], xdata[0], 0.1, max_val, ydata[-1])

    popt, *_ = scipy.optimize.curve_fit(function_, xdata, ydata, p0=p0, method='trf',
                                        bounds=(lower_bounds, upper_bounds))

    return Parameters(*popt, xdata[-1])


def fit_periodic_part(transition_index, end_index, single_particle_data):
    fft_out = np.fft.fft(single_particle_data['residuum'][transition_index: end_index])

    # fft_out[len(fft_out)//2:-1] = np.zeros(len(fft_out) - len(fft_out)//2 - 1)
    # fft_out[0:len(fft_out) // 2] = np.zeros(len(fft_out) // 2)
    main_freqs = np.argsort(fft_out)[-20:]
    main_amps = [fft_out[f] for f in main_freqs]

    fft_out = np.zeros(len(fft_out), dtype=complex)
    fft_out[main_freqs] = main_amps

    return fft_out


class Parameters:
    def __init__(self, unactivated_val, increase_point, increase_steepness, activated_val, decreased_val, end):
        # TODO: check parameters for validity
        self.unactivated_val = unactivated_val
        self.increase_point = increase_point
        self.increase_steepness = increase_steepness
        self.activated_val = activated_val
        self.decreased_val = decreased_val
        self.end = end

        alpha = 0.99
        tmp = ((1 - alpha) * self.activated_val) / (alpha * self.activated_val - self.unactivated_val)
        if tmp > 0:
            self.transition_point = - 1 / self.increase_steepness * \
                                    math.log(((1 - alpha) * self.activated_val) /
                                             (alpha * self.activated_val - self.unactivated_val)) + self.increase_point
        else:
            self.transition_point = None
        self.val_at_transition = function(self.transition_point, self)

    def list(self):
        return [self.unactivated_val, self.increase_point, self.increase_steepness, self.activated_val,
                self.decreased_val]

    def __str__(self):
        return f"unactivated_val: {self.unactivated_val}, increase_point: {self.increase_point}, " \
               f"increase_steepness: {self.increase_steepness}, activated_val: {self.activated_val}, " \
               f"decreased_val: {self.decreased_val}"


def function(t, parameters):
    if parameters.transition_point is None:     # transition point lies outside datapoints (flat left side)
        return parameters.unactivated_val
    elif t <= parameters.transition_point:      # logistic function before transition point
        L = parameters.activated_val - parameters.unactivated_val
        t_0 = parameters.increase_point
        k = parameters.increase_steepness
        tmp = -k * (t - t_0)
        if tmp <= 32:
            res = L / (1 + math.exp(tmp))
        else:
            res = 0
        res = res + parameters.unactivated_val
        return res
    else:                                       # linear decrease after transition point
        k = (parameters.decreased_val - parameters.val_at_transition) / (parameters.end - parameters.transition_point)
        return k * (t - parameters.transition_point) + parameters.val_at_transition


def function_(t_arr, unactivated_val, increase_point, increase_steepness, activated_val, decreased_val):
    p = Parameters(unactivated_val, increase_point, increase_steepness, activated_val, decreased_val, t_arr[-1])
    return [function(t, p) for t in t_arr]


def main(visualize=True):
    matplotlib.use('TkAgg')
    data = read_data()

    # visualize particle calcium concentration
    for particle_idx in set(data['particle']):
        single_particle_data = data.loc[data['particle'] == particle_idx][['frame', 'ratio']]
        xdata, ydata = [x for x in single_particle_data['frame']], [y for y in single_particle_data['ratio']]

        if len(xdata) < 10:     # skip if too short
            continue

        # calculate fit for basic function and visualize results
        par = fit_basic_function(xdata, ydata)
        single_particle_data['fitted'] = function_(xdata, *par.list())
        if visualize:
            visualize_single_particle(single_particle_data)

        # calculate residuum
        single_particle_data['residuum'] = single_particle_data['ratio'] - single_particle_data['fitted']

        # do fft to fit a periodic function into the residuum
        transition_index, end_index = int((np.abs(xdata - par.transition_point)).argmin()), len(xdata)
        fft_out = fit_periodic_part(transition_index, end_index, single_particle_data)
        single_particle_data['fft'] = np.concatenate((np.zeros(transition_index), np.real(np.fft.ifft(fft_out))))

        # visualize the fitted periodic function
        if visualize:
            ax = single_particle_data.plot.scatter(x="frame", y="residuum", xlim=(par.transition_point, par.end))
            single_particle_data.plot(x="frame", y="fft", xlim=(par.transition_point, par.end), ax=ax, color="red")
            plt.show()

        # visualize end result
        if visualize:
            visualize_single_particle(single_particle_data)


if __name__ == '__main__':
    main()
