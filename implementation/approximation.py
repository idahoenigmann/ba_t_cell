import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
import scipy


def read_data(file: str) -> pandas.DataFrame:
    """
    reads data from file
    :param file: name of file without .h5-suffix
    :return: pandas dataframe of t-cell calcium concentrations
    """
    return pandas.DataFrame(pd.read_hdf(f"../data/{file}/{file}.h5"))


def calc_transition_point(w: float, k: float, alpha: float = 0.99) -> float | None:
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


def sigmoid_and_linear_decreasing_(x, w1, t, w2, a, d, u, k1, k2):
    """
    piecewise function, logistic function followed by linear function
    :param x: point at which to evaluate the function
    :param w1: midpoint of increasing sigmoid function
    :param t: transition point between increasing sigmoid and decreasing sigmoid
    :param w2: midpoint of decreasing sigmoid function
    :param a: activated value, supremum of sigmoid function
    :param d: decreased value, value reached at end of decrease
    :param u: unactivated value, infimum of sigmoid function
    :param k1: steepness of increasing sigmoid function
    :param k2: steepness of decreasing sigmoid function
    """
    if t is None:  # transition point lies outside datapoints (flat left side)
        return u
    elif x <= t:  # logistic (increasing) function before transition point
        tmp = -k1 * (x - w1)
        if tmp <= 32:
            res = (a - u) / (1 + math.exp(tmp))
        else:
            res = 0
        return res + u
    else:  # logistic (decreasing) function after transition point
        tmp = -k2 * (x - w2)
        if tmp <= 32:
            res = (a - d) / (1 + math.exp(tmp))
        else:
            res = 0
        return res + d


def sigmoid_and_linear_decreasing(x_arr, w1, w2, a, d, u, k1, k2):
    """
    wrapper for sigmoid_and_linear_decreasing_ function, takes list as input
    :param x_arr: list of points at which to evaluate the function
    :param w1: midpoint of increasing sigmoid function
    :param w2: midpoint of decreasing sigmoid function
    :param a: activated value, supremum of sigmoid function
    :param d: decreased value, value reached at end of decrease
    :param u: unactivated value, infimum of sigmoid function
    :param k1: steepness of increasing sigmoid function
    :param k2: steepness of decreasing sigmoid function
    """
    transition_point = calc_transition_point(w1, k1)
    return np.vectorize(sigmoid_and_linear_decreasing_)(x_arr, w1, transition_point, w2, a, d, u, k1, k2)


def approximate_with_sigmoid_curve(dataframe: pandas.DataFrame) -> dict:
    """
    approximates the datapoints of dataframe with a combination of a sigmoid curve and a linear decreasing function
    changes dataframe! adds a column named fit_sigmoid
    :param dataframe: datapoints to approximate, must contain column ratio
    :returns: parameters for sigmoid curve
    """

    def sigmoid_and_linear_decreasing_for_approx(x_arr, w1_start, w2_w1, a_d, d_u, u, k1, k2):
        """
        wrapper for sigmoid_and_linear_decreasing_ function, takes list as input
        :param x_arr: list of points at which to evaluate the function
        :param w1: midpoint of increasing sigmoid function
        :param w2: midpoint of decreasing sigmoid function
        :param a: activated value, supremum of sigmoid function
        :param d: decreased value, value reached at end of decrease
        :param u: unactivated value, infimum of sigmoid function
        :param k1: steepness of increasing sigmoid function
        :param k2: steepness of decreasing sigmoid function
        """
        w1 = w1_start + start
        w2 = w2_w1 + w1
        d = d_u + u
        a = a_d + d
        transition_point = calc_transition_point(w1, k1)
        return np.vectorize(sigmoid_and_linear_decreasing_)(x_arr, w1, transition_point, w2, a, d, u, k1, k2)

    min_val, median_val, max_val = np.min(dataframe['ratio']), np.median(dataframe['ratio']), np.max(dataframe['ratio'])
    start, end = min(dataframe['frame']), max(dataframe['frame'])
    lower_bounds = (0,           0,           0,       0,       min_val, 0.05, -1)
    upper_bounds = (end - start, end - start, max_val, max_val, max_val, 3, -0.01)
    p0 = (0, (end - start)/2, max_val - median_val, median_val - min_val, min_val, 0.1, -0.03)

    popt, *_ = scipy.optimize.curve_fit(sigmoid_and_linear_decreasing_for_approx, dataframe['frame'], dataframe['ratio'], p0=p0,
                                        method='trf', bounds=(lower_bounds, upper_bounds))

    w1_start, w2_w1, a_d, d_u, u, k1, k2 = popt
    w1 = w1_start + start
    w2 = w2_w1 + w1
    d = d_u + u
    a = a_d + d
    t = calc_transition_point(w1, k1)
    dataframe['fit_sigmoid'] = sigmoid_and_linear_decreasing(dataframe['frame'], w1, w2, a, d, u, k1, k2)
    return {"start": start, "end": end, 'w1': w1, 't': t, 'w2': w2, 'e': end, 'a': a, 'd': d, 'u': u, 'k1': k1, 'k2': k2}


def freq_to_func(freqs, amps, start, end):
    # delete all but main frequencies
    fft_out = np.zeros(end - start, dtype=complex)
    fft_out[freqs] = amps

    return np.concatenate((np.zeros(start), np.real(np.fft.ifft(fft_out))))


def approximate_residuum_with_fft(dataframe: pandas.DataFrame, number_of_frequencies_kept: int, start: int, end: int)\
        -> dict:
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
    main_amps = [abs(fft_out[f]) for f in main_freqs]

    dataframe['fit_sin'] = freq_to_func(main_freqs, main_amps, start, end)

    freqs = {f'freq{i}': main_freqs[-i] for i in range(len(main_freqs))}
    amps = {f'amp{i}': abs(main_amps[-i]) for i in range(len(main_amps))}

    return {**freqs, **amps}


def visualize(dataframe: pandas.DataFrame, titel: str = "", return_fig: bool = False):
    """
    visualizes datapoints and (optional) approximations
    :param titel: title of plot
    :param dataframe: datapoints to visualize, must contain column ratio, can contain columns fit_sigmoid and fit_sin
    """
    fig, axes = plt.subplots(2, sharex=True)
    axes[0].title.set_text(titel)
    axes[0].set_ylim(0, 5)
    axes[0].set_xlim(0, 1000)

    axes[0].xaxis.label.set_size(12)
    axes[0].yaxis.label.set_size(12)
    axes[0].tick_params(axis='both', labelsize=12)
    axes[1].xaxis.label.set_size(12)
    axes[1].yaxis.label.set_size(12)
    axes[1].tick_params(axis='both', labelsize=12)

    for i in range(0, 2):
        dataframe.plot.scatter(x="frame", y="ratio", color="#000000", ax=axes[i])
        if 'fit_sigmoid' in dataframe.columns:
            dataframe.plot(x='frame', y='fit_sigmoid', color="#FF9904", ax=axes[i])
        if 'fit_total' in dataframe.columns:
            dataframe.plot(x='frame', y='fit_total', color="#E9190C", ax=axes[i])
    if return_fig:
        return fig
    else:
        plt.show()


def calc_residuum_and_error(dataframe: pandas.DataFrame) -> float:
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


def particle_to_parameters(particle_data: pandas.DataFrame, output_information: bool = True,
                           visualize_particles: bool = False, select_by_input: bool = False, titel: str = "") -> dict:
    """
    generates the parameters for a single particle
    :param titel: title shown in plot if visualization is turned on
    :param particle_data: data of a single particle
    :param output_information: if True information such as resulting parameters and error will be printed
    :param visualize_particles: if True a plot showing the data as well as approximation of each particle is shown
    :param select_by_input: if True the user is asked whether to confirm or discard the approximation for the particle
    :return: dictionary where keys are particle indices and values are corresponding parameter dictionaries
    """

    if select_by_input:
        visualize_particles = True

    # might throw error if best fit was not found within limited number of tries
    parameters_sigmoid = approximate_with_sigmoid_curve(particle_data)

    mse_sigmoid = calc_residuum_and_error(particle_data)

    # calculate the point at which the transition between sigmoid and linear function
    if parameters_sigmoid['t'] is None:
        transition_index = particle_data['frame'][-1]
    else:
        transition_index = int((np.abs(particle_data['frame'] - parameters_sigmoid['t'])).argmin())

    # use fft to fit sin
    parameters_sin = approximate_residuum_with_fft(particle_data, 10, transition_index,
                                                   len(particle_data['frame']))

    for i in range(10):
        if f"freq{i}" not in parameters_sin.keys():
            parameters_sin[f"freq{i}"] = 0
            parameters_sin[f"amp{i}"] = 0

    particle_parameters = {**parameters_sigmoid, **parameters_sin}
    particle_data['fit_total'] = particle_data['fit_sigmoid'] + particle_data['fit_sin']
    mse_total = calc_residuum_and_error(particle_data)

    particle_parameters["mse_sigmoid"] = mse_sigmoid
    particle_parameters["mse_total"] = mse_total

    if output_information:
        print(f"parameters: {particle_parameters}")
        print(f"mse sigmoid: {mse_sigmoid}, mse total: {mse_total}")
    if visualize_particles:
        visualize(particle_data, titel)

        if select_by_input:
            accepted = input('accept (y/n): ')
            if len(accepted) == 0 or accepted[0] not in 'yY':
                raise RuntimeError('Particle approximation was not accepted by user.')

    return particle_parameters


def main(file_name):
    # matplotlib.use('TkAgg')
    print(file_name)

    data = read_data(file_name)
    # filter out nan and inf values as well as too low and high values
    data = data[np.isfinite(data["ratio"])]
    data = data[np.less(data["ratio"], np.full((len(data["ratio"])), 5))]
    data = data[np.greater(data["ratio"], np.full((len(data["ratio"])), 0))]

    all_parameters = list()
    parameters_saved = ["idx", "start", "end", 's', 'w1', 't', 'w2', 'e', 'a', 'd', 'u', 'k1', 'k2', "mse_sigmoid",
                        "mse_total"]
    # TODO 10 is not a fixed value, but a parameter of approximate_residuum_with_fft
    parameters_saved = parameters_saved + [f"freq{i}" for i in range(10)] + [f"amp{i}" for i in range(10)]

    for particle_idx in set(data['particle']):
        # get data of a single particle
        single_particle_data = data.loc[data['particle'] == particle_idx][['frame', 'ratio']]

        # skip if too few datapoints
        if len(single_particle_data['frame']) < 300:
            continue

        try:  # throws error if no best fit was found or if particle was rejected by user (select_by_input)
            parameters = particle_to_parameters(single_particle_data, output_information=False,
                                                visualize_particles=False, select_by_input=False,
                                                titel=f"particle {particle_idx}")

            parameters["idx"] = particle_idx
            parameters['s'] = calc_transition_point(parameters['w1'], parameters['k1'], alpha=0.01)
            all_parameters.append([parameters[e] for e in parameters_saved])

        except Exception as e:
            print(e)
            continue

    np.savetxt(f"intermediate/particle_parameters_{file_name}.csv", np.matrix(np.array(all_parameters)), delimiter=',',
               newline='\n', header=",".join(parameters_saved))


if __name__ == '__main__':
    """
    Approximates data, visualizes results, saves parameters to file

    In this file you can change the minimum length the recordings must have to be processed, and the parameters that
    get written to the csv file.
    To generate all particle parameters and save them to a csv file set output_information and visualize_particles
    to False.
    """

    main("human_positive")
    main("human_negative")
    main("mouse_positive")
    main("mouse_negative")
