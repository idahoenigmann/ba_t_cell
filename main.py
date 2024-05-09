import pandas
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import math
import numpy as np
import scipy


def read_data():
    data = pd.read_hdf('data/SLB7_231218.h5')
    print(f"header: {[e for e in data.columns]}")

    # lengths = [len(data.loc[data['particle'] == idx][['frame']]) for idx in set(data['particle'])]
    # print(len([lengths[i] for i in range(len(lengths)) if lengths[i] == 948]))
    # print(len([lengths[i] for i in range(len(lengths)) if lengths[i] != 948]))
    # print(len([lengths[i] for i in range(len(lengths)) if lengths[i] >= 100]))
    return data


def visualize_single_particle(single_particle_data, ax=None):
    if ax is None:
        single_particle_data.plot.scatter(x="frame", y="ratio")
    else:
        single_particle_data.plot.scatter(x="frame", y="ratio", ax=ax)
    plt.show()


def visualize_data():
    matplotlib.use('TkAgg')
    data = read_data()

    # visualize particle calcium concentration
    for particle_idx in set(data['particle']):
        single_particle_data = data.loc[data['particle'] == particle_idx][['frame', 'ratio']]

        xdata, ydata = [x for x in single_particle_data['frame']], [y for y in single_particle_data['ratio']]

        if len(xdata) < 10:
            continue

        min_val, median_val, max_val = np.min(ydata), np.median(ydata), np.max(ydata)
        lower_bounds = (min_val, xdata[0], 0.05, median_val, min_val)
        upper_bounds = (median_val, xdata[-1], 10, max_val, max_val)
        p0 = [(lower_bounds[i] + upper_bounds[i])/2 for i in range(len(lower_bounds))]
        p0[1], p0[2] = lower_bounds[1], 0.1

        popt, *_ = scipy.optimize.curve_fit(function_, xdata, ydata, p0=p0, method='trf',
                                            bounds=(lower_bounds, upper_bounds))

        f = pandas.DataFrame({'frame': xdata, 'data': function_(xdata, *popt)})
        ax = f.plot(x='frame', y='data', color="red")
        visualize_single_particle(single_particle_data, ax)


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
    if parameters.transition_point is None:
        return parameters.unactivated_val
    # logistic function
    if t <= parameters.transition_point:
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
    else:
        k = (parameters.decreased_val - parameters.val_at_transition) / (parameters.end - parameters.transition_point)
        return k * (t - parameters.transition_point) + parameters.val_at_transition


def function_(t_arr, unactivated_val, increase_point, increase_steepness, activated_val, decreased_val):
    p = Parameters(unactivated_val, increase_point, increase_steepness, activated_val, decreased_val, t_arr[-1])
    return [function(t, p) for t in t_arr]


if __name__ == '__main__':
    visualize_data()
