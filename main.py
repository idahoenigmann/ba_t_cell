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
            res = res + u
            return res
        else:  # linear decrease after transition point
            val_at_transition = sigmoid_and_linear_decreasing_(t, w, t, a, d, u, k)
            k = (d - val_at_transition) / (end - t)
            return k * (x - t) + val_at_transition

    def sigmoid_and_linear_decreasing(x_arr, w, a, d, u, k):
        alpha = 0.99
        t = - 1 / k * math.log(((1 - alpha) * a) / (alpha * a - u)) + w
        f = np.vectorize(sigmoid_and_linear_decreasing_)
        # [sigmoid_and_linear_decreasing_(x, w, t, e, a, d, u, k) for x in x_arr]
        return f(x_arr, w, t, a, d, u, k)

    min_val, median_val, max_val = np.min(dataframe['ratio']), np.median(dataframe['ratio']), np.max(dataframe['ratio'])
    start, end = min(dataframe['frame']), max(dataframe['frame'])
    lower_bounds = (start, median_val, min_val, min_val, 0.05)
    upper_bounds = (end, max_val, max_val, median_val, 10)
    p0 = (start, max_val, median_val, min_val, 0.1)

    popt, *_ = scipy.optimize.curve_fit(sigmoid_and_linear_decreasing, dataframe['frame'], dataframe['ratio'], p0=p0,
                                        method='trf', bounds=(lower_bounds, upper_bounds))
    dataframe['fit_sigmoid'] = sigmoid_and_linear_decreasing(dataframe['frame'], *popt)
    return popt


def approximate_residuum_with_sin(dataframe):
    """
    approximates the residuum by a sin function with fluctuating amplitude
    changes dataframe! adds a column named fit_sin
    :param dataframe: datapoints to approximate, must contain columns ratio and fit_sigmoid
    :returns: parameters for sin
    """
    return 0


def visualize(dataframe):
    """
    visualizes datapoints and (optional) approximations
    :param dataframe: datapoints to visualize, must contain column ratio, can contain columns fit_sigmoid and fit_sin
    """
    ax = single_particle_data.plot.scatter(x="frame", y="ratio")
    if 'fit_sigmoid' in single_particle_data.columns:
        if 'fit_sin' in single_particle_data.columns:
            pass
        else:
            single_particle_data.plot(x='frame', y='fit_sigmoid', color="red", ax=ax)
    plt.show()


if __name__ == '__main__':
    matplotlib.use('TkAgg')
    data = read_data()

    for particle_idx in set(data['particle']):
        single_particle_data = data.loc[data['particle'] == particle_idx][['frame', 'ratio']]

        approximate_with_sigmoid_curve(single_particle_data)

        visualize(single_particle_data)
