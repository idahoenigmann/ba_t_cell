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


def approximate_with_sigmoid_curve(dataframe):
    """
    approximates the datapoints of dataframe with a combination of a sigmoid curve and a linear decreasing function
    changes dataframe! adds a column named fit_sigmoid
    :param dataframe: datapoints to approximate, must contain column ratio
    :returns: parameters for sigmoid curve
    """

    def calc_transition_point(w, a, u, k, alpha=0.99):
        """
        calculate transition point as point where supremum is almost reached
        :param w, a, u, k: function parameters
        :param alpha: margin to supremum
        :return: transition point
        """
        return - 1 / k * math.log(((1 - alpha) * a) / (alpha * a - u)) + w

    def sigmoid_and_linear_decreasing_(x, w, t, a, d, u, k):
        """
        w   midpoint of sigmoid function
        t   transition point between sigmoid and linear function
        e   end of x-axis
        a   activated value, supremum of sigmoid function
        d   decreased value, value reached at end of decrease
        u   unactivated value, infimum of sigmoid function
        k   steepness of sigmoid function
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
        """
        t = calc_transition_point(w, a, u, k)
        return np.vectorize(sigmoid_and_linear_decreasing_)(x_arr, w, t, a, d, u, k)

    min_val, median_val, max_val = np.min(dataframe['ratio']), np.median(dataframe['ratio']), np.max(dataframe['ratio'])
    start, end = min(dataframe['frame']), max(dataframe['frame'])
    lower_bounds = (start, median_val, min_val, min_val, 0.05)
    upper_bounds = (end, max_val, max_val, median_val, 10)
    p0 = (start, max_val, median_val, min_val, 0.1)

    popt, *_ = scipy.optimize.curve_fit(sigmoid_and_linear_decreasing, dataframe['frame'], dataframe['ratio'], p0=p0,
                                        method='trf', bounds=(lower_bounds, upper_bounds))
    dataframe['fit_sigmoid'] = sigmoid_and_linear_decreasing(dataframe['frame'], *popt)

    w, a, d, u, k = popt
    t = calc_transition_point(w, a, u, k)
    return {'w': w, 't': t, 'e': end, 'a': a, 'd': d, 'u': u, 'k': k}


def approximate_residuum_with_sin(dataframe, start, end):
    """
    approximates the residuum by a sin function with fluctuating amplitude
    changes dataframe! adds a column named fit_sin
    :param dataframe: datapoints to approximate, must contain columns ratio and fit_sigmoid
    :returns: parameters for sin
    """

    def sin_with_changing_amplitude_(x, f, phi, a_0):
        """
        f     frequency
        phi   phase
        a_i   coefficients of amplitude polynomial of degree 0
        """
        return (a_0) * math.sin(2 * math.pi * f * x + phi)

    def sin_with_changing_amplitude(x_arr, f, phi, a_0):
        """
        wrapper for sin_with_changing_amplitude_ function, takes list as input
        """
        return np.vectorize(sin_with_changing_amplitude_)(x_arr, f, phi, a_0)

    lower_bounds = (0.001, 0, -np.inf)
    upper_bounds = (0.1, 2 * math.pi, np.inf)
    p0 = (0.01, 0, 0)

    popt, *_ = scipy.optimize.curve_fit(sin_with_changing_amplitude, dataframe['frame'][start:end],
                                        dataframe['residuum'][start:end], p0=p0,
                                        method='trf', bounds=(lower_bounds, upper_bounds))

    dataframe['fit_sin'] = np.concatenate(
        (np.zeros(start), sin_with_changing_amplitude(dataframe['frame'][start:end], *p0)))

    f, phi, a_0 = popt
    return {'f': f, 'phi': phi, 'a_0': a_0}


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


if __name__ == '__main__':
    matplotlib.use('TkAgg')
    data = read_data()

    for particle_idx in set(data['particle']):
        single_particle_data = data.loc[data['particle'] == particle_idx][['frame', 'ratio']]

        parameters = approximate_with_sigmoid_curve(single_particle_data)
        single_particle_data['residuum'] = single_particle_data['ratio'] - single_particle_data['fit_sigmoid']

        transition_index = int((np.abs(single_particle_data['frame'] - parameters['t'])).argmin())

        parameters = {**parameters, **approximate_residuum_with_sin(single_particle_data, transition_index,
                                                                    len(single_particle_data['frame']))}
        single_particle_data['fit_total'] = single_particle_data['fit_sigmoid'] + single_particle_data['fit_sin']

        print(parameters)
        visualize(single_particle_data)
