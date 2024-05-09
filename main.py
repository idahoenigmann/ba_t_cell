import pandas
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import math
import numpy as np
import scipy


def read_data():
    data = pd.read_hdf('../SLB7_231218.h5')
    print(f"header: {[e for e in data.columns]}")

    lengths = [len(data.loc[data['particle'] == idx][['frame']]) for idx in set(data['particle'])]
    # print(len([lengths[i] for i in range(len(lengths)) if lengths[i] == 948]))
    # print(len([lengths[i] for i in range(len(lengths)) if lengths[i] != 948]))
    # print(len([lengths[i] for i in range(len(lengths)) if lengths[i] >= 100]))
    return data


def visualize_single_particle(particle_idx, data):
    single_particle_data = data.loc[data['particle'] == particle_idx][['frame', 'ratio']]
    single_particle_data.plot.scatter(x="frame", y="ratio")
    plt.show()


def visualize_data():
    matplotlib.use('TkAgg')

    data = read_data()

    # visualize particle calcium concentration
    for particle_idx in set(data['particle']):
        visualize_single_particle(particle_idx, data)


class Parameters:
    def __init__(self, unactivated_val, start_increase, increase, end_increase, activated_val, osc_dec_amp,
                 osc_dec_freq, osc_dec_offset, end_decrease, decreased_val, osc_end_amp, osc_end_freq, osc_end_offset):
        # TODO: check parameters for validity
        self.unactivated_val = unactivated_val
        self.start_increase = start_increase
        self.increase = increase
        self.end_increase = end_increase
        self.activated_val = activated_val
        self.osc_dec_amp = osc_dec_amp
        self.osc_dec_freq = osc_dec_freq
        self.osc_dec_offset = osc_dec_offset
        self.end_decrease = end_decrease
        self.decreased_val = decreased_val
        self.osc_end_amp = osc_end_amp
        self.osc_end_freq = osc_end_freq
        self.osc_end_offset = osc_end_offset

    def list(self):
        return [self.unactivated_val, self.start_increase, self.increase, self.end_increase, self.activated_val,
                self.osc_dec_amp, self.osc_dec_freq, self.osc_dec_offset, self.end_decrease, self.decreased_val,
                self.osc_end_amp, self.osc_end_freq, self.osc_end_offset]

    def __str__(self):
        return f"unactivated_val: {self.unactivated_val}, start_increase: {self.start_increase}, " \
               f"increase: {self.increase}, end_increase: {self.end_increase}, activated_val: {self.activated_val}, " \
               f"osc_dec_amp: {self.osc_dec_amp}, osc_dec_freq: {self.osc_dec_freq}, " \
               f"osc_dec_offset: {self.osc_dec_offset}, end_decrease: {self.end_decrease}, " \
               f"decreased_val: {self.decreased_val}, osc_end_amp: {self.osc_end_amp}, " \
               f"osc_end_frequ: {self.osc_end_freq}, osc_end_offset: {self.osc_end_offset}"


def function(t, parameters):
    if t < parameters.start_increase:
        return parameters.unactivated_val
    elif t < parameters.end_increase:
        t = t - (parameters.end_increase + parameters.start_increase) / 2
        k = parameters.increase
        b_0 = parameters.unactivated_val
        s = parameters.activated_val - parameters.unactivated_val
        return b_0 + s / (1 + math.exp(-k * s * t) * (s / b_0 - 1))
    elif t < parameters.end_decrease:
        t = t - parameters.end_increase
        k = (parameters.decreased_val - parameters.activated_val) / (parameters.end_decrease - parameters.end_increase)
        val = parameters.activated_val + k * t
        val = val + parameters.osc_dec_amp * math.sin(parameters.osc_dec_freq * (t - parameters.osc_dec_offset))
        return max(val, 0)
    elif t > parameters.end_decrease:
        val = parameters.decreased_val
        val = val + parameters.osc_end_amp * math.sin(parameters.osc_end_freq * (t - parameters.osc_end_offset))
        return max(val, 0)


def function_(t_arr, unactivated_val, start_increase, increase, end_increase, activated_val, osc_dec_amp,
              osc_dec_freq, osc_dec_offset, end_decrease, decreased_val, osc_end_amp, osc_end_freq, osc_end_offset):
    parameters = Parameters(unactivated_val, start_increase, increase, end_increase, activated_val, osc_dec_amp,
                            osc_dec_freq, osc_dec_offset, end_decrease, decreased_val, osc_end_amp, osc_end_freq,
                            osc_end_offset)
    return [function(t, parameters) for t in t_arr]


def find_start_increase(ydata, thres=5):
    a = [1 if (ydata[i + 1] - ydata[i]) > 0 else 0 for i in range(len(ydata) - 1)]
    return [sum(a[i:i + thres]) for i in range(len(a) - thres)].index(thres)


def find_end_increase(ydata, thres=5):
    a = [np.average(ydata[i:i + thres]) for i in range(len(ydata) - thres)]
    end_increase = max(enumerate(a), key=lambda x: x[1])[0]
    return end_increase, a[end_increase]


def find_end_decrease(ydata):
    return find_start_increase(ydata[::-1], 1)


def optimize(xdata, ydata):
    p = Parameters(unactivated_val=50, start_increase=100, increase=0.1, end_increase=40, activated_val=100,
                   osc_dec_amp=1, osc_dec_freq=1, osc_dec_offset=math.pi/2, end_decrease=120, decreased_val=80,
                   osc_end_amp=1, osc_end_freq=1, osc_end_offset=math.pi/2)

    # increase, osc_dec_freq, osc_dec_offset, end_decrease, decreased_val, osc_end_amp, osc_end_freq, osc_end_offset

    start_increase = find_start_increase(ydata)
    p.start_increase = xdata[start_increase]
    p.unactivated_val = np.average(ydata[0:start_increase])

    end_increase, activated_val = find_end_increase(ydata)
    p.end_increase = xdata[end_increase]
    p.activated_val = activated_val

    p.osc_dec_amp = 0
    end_decrease = find_end_decrease(ydata)
    p.end_decrease = xdata[end_decrease]
    p.decreased_val = ydata[end_decrease]
    p.osc_end_amp = 0

    return p


if __name__ == '__main__':
    visualize_data()
    matplotlib.use('TkAgg')
    data = read_data()

    single_particle_data = data.loc[data['particle'] == 1][['frame', 'ratio']]
    xdata = [x for x in single_particle_data['frame']]
    ydata = [y for y in single_particle_data['ratio']]

    parameters = optimize(xdata, ydata)
    print(parameters)

    # parameters = Parameters(unactivated_val=50, start_increase=100, increase=0.001, end_increase=40, activated_val=100,
    #                        osc_dec_amp=1, osc_dec_freq=1, osc_dec_offset=0, end_decrease=120, decreased_val=80,
    #                        osc_end_amp=1, osc_end_freq=1, osc_end_offset=0)
    # popt, pcov = scipy.optimize.curve_fit(function_, xdata, ydata, p0=[parameters.list()])
    # parameters = Parameters(*popt)
    f = pandas.DataFrame({'frame': xdata,
                          'data': [function(x, parameters) for x in xdata]})
    f.plot.scatter(x='frame', y='data')
    visualize_single_particle(1, data)
