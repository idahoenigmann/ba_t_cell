import math
import scipy
import numpy as np
import pandas
from sklearn.mixture import GaussianMixture
import pandas as pd


def approximate(dataframe):
    def calc_t(w, k, alpha = 0.99):
        tmp = 1 / alpha - 1
        if tmp < 0.0001:
            return None
        try:
            return w - math.log(tmp) / k
        except ValueError:
            return None

    def approx_func_(x, w1, t, w2, a, d, u, k1, k2):
        if t is None:  # transition point lies outside datapoints
            return u
        elif x <= t:  # logistic function before transition point
            tmp = -k1 * (x - w1)
            if tmp <= 32:
                res = (a - u) / (1 + math.exp(tmp))
            else:
                res = 0
            return res + u
        else:  # logistic function after transition point
            tmp = -k2 * (x - w2)
            if tmp <= 32:
                res = (a - d) / (1 + math.exp(tmp))
            else:
                res = 0
            return res + d

    def approx_func(x_arr, w1_start, w2_w1, a_d, d_u, u, k1, k2):
        w1 = w1_start + start
        w2 = w2_w1 + w1
        d = d_u + u
        a = a_d + d
        transition_point = calc_t(w1, k1)
        return (np.vectorize(approx_func_)
                (x_arr, w1, transition_point, w2, a, d, u, k1, k2))

    min_val = np.min(dataframe['ratio'])
    median_val = np.median(dataframe['ratio'])
    max_val = np.max(dataframe['ratio'])
    start, end = min(dataframe['frame']), max(dataframe['frame'])

    lower_bounds = (0, 0, 0, 0, min_val, 0.05, -1)
    upper_bounds = (end-start, end-start, max_val, max_val,
                    max_val, 3, -0.01)
    p0 = (0, (end-start)/2, max_val-median_val, median_val-min_val,
          min_val, 0.1, -0.03)

    popt, *_ = scipy.optimize.curve_fit(approx_func, dataframe['frame'],
                                        dataframe['ratio'], p0=p0,
                                        method='trf',
                                        bounds=(lower_bounds, upper_bounds))

    w1_start, w2_w1, a_d, d_u, u, k1, k2 = popt
    w1 = w1_start + start
    w2 = w2_w1 + w1
    d = d_u + u
    a = a_d + d
    t = calc_t(w1, k1)
    return {"start": start, "end": end, 'w1': w1, 't': t, 'w2': w2,
            'e': end, 'a': a, 'd': d, 'u': u, 'k1': k1, 'k2': k2}


def approximation_loop(file_name):

    # read data
    data = pandas.DataFrame(pd.read_hdf(f"../data/{file_name}"))

    # filter data
    data = data[np.isfinite(data["ratio"])]
    data = data[np.less(data["ratio"], np.full((len(data["ratio"])), 5))]
    data = data[np.greater(data["ratio"], np.full((len(data["ratio"])), 0))]

    all_parameters = list()
    parameters_saved = ["idx", "start", "end", 'w1', 't', 'w2',
                        'a', 'd', 'u', 'k1', 'k2']

    for particle_idx in set(data['particle']):
        # get data of a single particle
        single_particle_data = (
            data.loc)[data['particle'] == particle_idx][['frame', 'ratio']]

        # skip if too few datapoints
        if len(single_particle_data['frame']) < 300:
            continue

        try:  # throws error if no best fit was found
            parameters = approximate(single_particle_data)

            parameters["idx"] = particle_idx
            all_parameters.append([parameters[e] for e in parameters_saved])

        except Exception as e:
            print(f"error in particle {particle_idx}: {e}")

    return all_parameters


def separate(neg_par, pos_par):
    prediction_parameters = ["a", "u", "d", "k1", "k2", "w1", "w2"]

    df_neg = pd.DataFrame(data=neg_par,
                          columns=["idx", "start", "end", 'w1', 't', 'w2',
                                   'a', 'd', 'u', 'k1', 'k2'])
    df_pos = pd.DataFrame(data=pos_par,
                          columns=["idx", "start", "end", 'w1', 't', 'w2',
                                   'a', 'd', 'u', 'k1', 'k2'])
    df_neg["activation"] = "negative"
    df_pos["activation"] = "negative"
    data = pd.concat([df_neg, df_pos])

    # clustering
    gm = GaussianMixture(n_components=2, covariance_type="full", n_init=10)
    gm.fit(data[prediction_parameters])
    data["predicted_clusters"] = gm.predict(data[prediction_parameters])

    # find association between predicted clusters and files
    neg_0 = len(data[(data['predicted_clusters'] == 0) &
                     (data['activation'] == "negative")])
    neg_1 = len(data[(data['predicted_clusters'] == 1) &
                     (data['activation'] == "negative")])
    pos_0 = len(data[(data['predicted_clusters'] == 0) &
                     (data['activation'] == "positive")])
    pos_1 = len(data[(data['predicted_clusters'] == 1) &
                     (data['activation'] == "positive")])

    permutation = (0, 1) if neg_0 + pos_1 > neg_1 + pos_0 else (1, 0)

    return permutation, gm


if __name__ == "__main__":
    FILE_NAME_NEG_CONTROL = "human_negative/human_negative.h5"
    FILE_NAME_POS_CONTROL = "human_positive/human_positive.h5"
    FILE_NAME_EXPERIMENT = "human_negative/human_negative.h5"

    neg_con_par = approximation_loop(FILE_NAME_NEG_CONTROL)
    pos_con_par = approximation_loop(FILE_NAME_POS_CONTROL)
    experiment_par = approximation_loop(FILE_NAME_EXPERIMENT)

    # filter out outliers

    per, gm = separate(neg_con_par, pos_con_par)

    df_exp = pd.DataFrame(data=experiment_par,
                          columns=["idx", "start", "end", 'w1', 't', 'w2',
                                   'a', 'd', 'u', 'k1', 'k2'])
    df_exp["predicted_clusters"] = gm.predict(df_exp[["a", "u", "d", "k1",
                                                      "k2", "w1", "w2"]])

    print(per)
    print(len(df_exp))
    print(len(df_exp[df_exp['predicted_clusters'] == 0]))
    print(len(df_exp[df_exp['predicted_clusters'] == 1]))
