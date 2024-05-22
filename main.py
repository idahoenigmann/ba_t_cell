import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy


def read_data():
    """
    reads data from file
    :return: pandas dataframe of t-cell calcium concentrations
    """
    return pd.read_hdf('data/SLB7_231218.h5')


def calc_transition_point(w, k, alpha=0.99):
    """
    calculate transition point as point where supremum is almost reached
    :param w: function parameter, middle point of increase
    :param k: function parameter, steepness of increase
    :param alpha: margin to supremum, e.g. 0.99 will return the point at which
    :return: transition point
    """
    tmp = 1 / alpha - 1
    if tmp < 0.0001:
        return None
    try:
        return w - math.log(tmp) / k
    except ValueError:
        return None


def approximate_with_sigmoid_curve(dataframe):
    """
    approximates the datapoints of dataframe with a combination of a sigmoid curve and a linear decreasing function
    changes dataframe! adds a column named fit_sigmoid
    :param dataframe: datapoints to approximate, must contain column ratio
    :returns: parameters for sigmoid curve
    """

    def sigmoid_and_linear_decreasing_(x, w, t, a, d, u, k):
        """
        piecewise function, logistic function followed by linear function
        :param x: point at which to evaluate the function
        :param w: midpoint of sigmoid function
        :param t: transition point between sigmoid and linear function
        :param a: activated value, supremum of sigmoid function
        :param d: decreased value, value reached at end of decrease
        :param u: unactivated value, infimum of sigmoid function
        :param k: steepness of sigmoid function
        """
        if t is None:  # transition point lies outside datapoints (flat left side)
            return u
        elif x <= t:  # logistic function before transition point
            tmp = -k * (x - w)
            if tmp <= 32:
                res = (a - u) / (1 + math.exp(tmp))
            else:
                res = 0
            return res + u
        else:  # linear decrease after transition point
            val_at_transition = sigmoid_and_linear_decreasing_(t, w, t, a, d, u, k)
            return (d - val_at_transition) / (end - t) * (x - t) + val_at_transition

    def sigmoid_and_linear_decreasing(x_arr, w, a, d, u, k):
        """
        wrapper for sigmoid_and_linear_decreasing_ function, takes list as input
        :param x_arr: list of points at which to evaluate the function
        :param w: midpoint of sigmoid function
        :param a: activated value, supremum of sigmoid function
        :param d: decreased value, value reached at end of decrease
        :param u: unactivated value, infimum of sigmoid function
        :param k: steepness of sigmoid function
        """
        t = calc_transition_point(w, k)
        return np.vectorize(sigmoid_and_linear_decreasing_)(x_arr, w, t, a, d, u, k)

    min_val, median_val, max_val = np.min(dataframe['ratio']), np.median(dataframe['ratio']), np.max(dataframe['ratio'])
    start, end = min(dataframe['frame']), max(dataframe['frame'])
    lower_bounds = (start, median_val + 0.002, min_val, min_val, 0.05)
    upper_bounds = (end, max_val, max_val, median_val - 0.002, 10)
    p0 = (start, max_val, median_val, min_val, 0.1)

    popt, *_ = scipy.optimize.curve_fit(sigmoid_and_linear_decreasing, dataframe['frame'], dataframe['ratio'], p0=p0,
                                        method='trf', bounds=(lower_bounds, upper_bounds))
    dataframe['fit_sigmoid'] = sigmoid_and_linear_decreasing(dataframe['frame'], *popt)

    w, a, d, u, k = popt
    t = calc_transition_point(w, k)
    return {'w': w, 't': t, 'e': end, 'a': a, 'd': d, 'u': u, 'k': k}


def approximate_residuum_with_fft(dataframe, number_of_frequencies_kept, start, end):
    """
    approximates the residuum between frames start and end by sin generated with fft
    changes dataframe! adds a column named fit_sin
    :param dataframe: datapoints to approximate, must contain columns ratio and fit_sigmoid
    :param number_of_frequencies_kept: how many frequencies are used in the approximation
    :param start: the point from which the approximation with periodic functions is to start
    :param end: the point from which the approximation with periodic functions is to end
    :returns: parameters for fft
    """

    # assumes frames are evenly spaced!
    fft_out = np.fft.fft(dataframe['residuum'][start:end])

    # get frequencies with the highest amplitude
    main_freqs = np.argsort(fft_out)[-number_of_frequencies_kept:]
    main_amps = [fft_out[f] for f in main_freqs]

    # delete all but main frequencies
    fft_out = np.zeros(end - start, dtype=complex)
    fft_out[main_freqs] = main_amps

    dataframe['fit_sin'] = np.concatenate((np.zeros(start), np.real(np.fft.ifft(fft_out))))

    return {'fft': dict([(main_freqs[i], main_amps[i]) for i in range(len(main_freqs))])}


def visualize(dataframe):
    """
    visualizes datapoints and (optional) approximations
    :param dataframe: datapoints to visualize, must contain column ratio, can contain columns fit_sigmoid and fit_sin
    """
    ax = dataframe.plot.scatter(x="frame", y="ratio")
    if 'fit_total' in dataframe.columns:
        dataframe.plot(x='frame', y='fit_total', color="red", ax=ax)
    elif 'fit_sigmoid' in dataframe.columns:
        dataframe.plot(x='frame', y='fit_sigmoid', color="red", ax=ax)
    plt.show()


def calc_residuum_and_error(dataframe):
    """
    calculates residuum and mean squared error
    :return: mean squared error of eiter fit_sigmoid or fit_total
    :param dataframe: data, must contain column fit_sigmoid or fit_total
    """
    if 'fit_total' in dataframe.columns:
        col_name = 'fit_total'
    elif 'fit_sigmoid' in dataframe.columns:
        col_name = 'fit_sigmoid'
    else:
        raise ValueError('dataframe does not contain one of the columns fit_sigmoid or fit_total.')
    dataframe['residuum'] = dataframe['ratio'] - dataframe[col_name]
    return np.dot(dataframe['residuum'], dataframe['residuum']) / len(dataframe['residuum'])


def all_particles_to_parameters(min_length_recording, output_information=True, visualize_particles=False):
    """
    generates the parameters for all particles of the dataset
    :param min_length_recording: all particles which were recorded for less than this number of frames will be skipped
    :param output_information: if True information such as resulting parameters and error will be printed
    :param visualize_particles: if True a plot showing the data as well as approximation of each particle is shown
    :return: dictionary where keys are particle indices and values are corresponding parameter dictionaries
    """
    if visualize_particles:
        matplotlib.use('TkAgg')
    data = read_data()

    all_parameters = dict()

    for particle_idx in set(data['particle']):
        # get data of a single particle
        single_particle_data = data.loc[data['particle'] == particle_idx][['frame', 'ratio']]

        # skip if too few datapoints
        if len(single_particle_data['frame']) < min_length_recording:
            continue

        # might throw error if best fit was not found within limited number of tries
        try:
            parameters_sigmoid = approximate_with_sigmoid_curve(single_particle_data)
        except RuntimeError as e:
            if output_information:
                print(e)
            continue

        rel_error_sigmoid = calc_residuum_and_error(single_particle_data)

        """
        # calculate the point at which the transition between sigmoid and linear function
        if parameters_sigmoid['t'] is None:
            transition_index = single_particle_data['frame'][-1]
        else:
            transition_index = int((np.abs(single_particle_data['frame'] - parameters_sigmoid['t'])).argmin())

        # use fft to fit sin
        parameters_sin = approximate_residuum_with_fft(single_particle_data, 10, transition_index,
                                                       len(single_particle_data['frame']))

        parameters = {**parameters_sigmoid, **parameters_sin}
        single_particle_data['fit_total'] = single_particle_data['fit_sigmoid'] + single_particle_data['fit_sin']
        rel_error_total = calc_residuum_and_error(single_particle_data)"""


        parameters, rel_error_total = parameters_sigmoid, 0
        all_parameters[particle_idx] = parameters

        if output_information:
            print(f"parameters: {parameters}")
            print(f"mse sigmoid: {rel_error_sigmoid}, mse total: {rel_error_total}")
        if visualize_particles:
            visualize(single_particle_data)

    return all_parameters


if __name__ == '__main__':
    all_parameters = all_particles_to_parameters(20, visualize_particles=True)
