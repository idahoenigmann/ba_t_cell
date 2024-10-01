import math
import scipy
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def approximate(dataframe):
    def calc_t(w, k, alpha=0.99):
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
    upper_bounds = (end - start, end - start, max_val, max_val,
                    max_val, 3, -0.01)
    p0 = (0, (end - start) / 2, max_val - median_val, median_val - min_val,
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
    return {"a": a, "u": u, "d": d, "k1": k1, "k2": k2, "w1": w1, "w2": w2}


def approximation_loop(file_name):
    data = pd.DataFrame(pd.read_hdf(f"../data/{file_name}"))

    # filter data
    data = data[np.isfinite(data["ratio"])]
    data = data[np.less(data["ratio"], np.full((len(data["ratio"])), 5))]
    data = data[np.greater(data["ratio"], np.full((len(data["ratio"])), 0))]

    all_parameters = list()
    parameters_saved = ["idx", "a", "u", "d", "k1", "k2", "w1", "w2"]

    for particle_idx in set(data['particle']):
        single_particle_data = (
            data.loc)[data['particle'] == particle_idx][['frame', 'ratio']]

        # skip if too few datapoints
        if len(single_particle_data['frame']) < 300:
            continue

        try:  # throws error if no best fit was found
            parameters = approximate(single_particle_data)

            parameters["idx"] = str(particle_idx) + file_name
            all_parameters.append([parameters[e] for e in parameters_saved])

        except Exception as e:
            print(f"error in particle {particle_idx}: {e}")

    return pd.DataFrame(all_parameters, columns=parameters_saved)


def remove_outliers(data, width, par_used):
    for par in set(par_used):
        mean, std = data[par].mean(), data[par].std()
        data = data.drop(data[(data[par] <= mean - std * width) &
                              (data[par] >= mean + std * width)].index)
    return data


def normalize(neg_df, pos_df, exp_df, normalized_columns):
    all_data = pd.concat([neg_df, pos_df, exp_df])

    scaler = StandardScaler()
    all_data[normalized_columns] = (scaler.fit_transform(
        all_data[normalized_columns]))

    neg_df = all_data[all_data["idx"].isin(neg_df["idx"])]
    pos_df = all_data[all_data["idx"].isin(pos_df["idx"])]
    exp_df = all_data[all_data["idx"].isin(exp_df["idx"])]

    return neg_df, pos_df, exp_df


def separate(neg_df, pos_df, prediction_parameters, clustering_method):
    neg_df["activation"] = "negative"
    pos_df["activation"] = "positive"
    data = pd.concat([neg_df, pos_df])

    # clustering
    if clustering_method == "gaussian_mixture":
        clustering = GaussianMixture(n_components=2, covariance_type="diag",
                                     n_init=10)
    elif clustering_method == "kmeans":
        clustering = KMeans(n_clusters=2, n_init=10)
    else:
        raise RuntimeError(f"Value of CLUSTERING_METHOD set to "
                           f"{clustering_method}, which is not one of "
                           f"[gaussian_mixture, kmeans].")
    data["predicted"] = clustering.fit_predict(data[prediction_parameters])

    # find association between predicted clusters and files
    neg_0 = len(data[(data['predicted'] == 0) &
                     (data['activation'] == "negative")])
    neg_1 = len(data[(data['predicted'] == 1) &
                     (data['activation'] == "negative")])
    pos_0 = len(data[(data['predicted'] == 0) &
                     (data['activation'] == "positive")])
    pos_1 = len(data[(data['predicted'] == 1) &
                     (data['activation'] == "positive")])

    permutation = (0, 1) if neg_0 + pos_1 > neg_1 + pos_0 else (1, 0)

    return permutation, clustering


if __name__ == "__main__":
    FILE_NAME_NEG_CONTROL = "human_negative/human_negative.h5"
    FILE_NAME_POS_CONTROL = "human_positive/human_positive.h5"
    FILE_NAME_EXPERIMENT = "human_negative/human_negative.h5"

    USED_COLUMNS = ["a", "u", "d", "k1", "k2", "w1", "w2"]
    CLUSTERING_METHOD = "gaussian_mixture"   #"gaussian_mixture" or "kmeans"

    neg_df = approximation_loop(FILE_NAME_NEG_CONTROL)
    pos_df = approximation_loop(FILE_NAME_POS_CONTROL)
    exp_df = approximation_loop(FILE_NAME_EXPERIMENT)

    n = min(len(neg_df), len(pos_df))
    neg_df, pos_df = neg_df.sample(n), pos_df.sample(n)

    neg_df, pos_df, exp_df = normalize(neg_df, pos_df, exp_df, USED_COLUMNS)

    neg_df = remove_outliers(neg_df, 3, USED_COLUMNS)
    pos_df = remove_outliers(pos_df, 3, USED_COLUMNS)
    exp_df = remove_outliers(exp_df, 3, USED_COLUMNS)

    per, clustering = separate(neg_df, pos_df, USED_COLUMNS,
                               CLUSTERING_METHOD)

    exp_df["predicted"] = clustering.predict(exp_df[USED_COLUMNS])

    print(f"positive: {len(exp_df[exp_df['predicted'] == per[1]])} / "
          f"{len(exp_df)}")
    print(f"negative: {len(exp_df[exp_df['predicted'] == per[0]])} / "
          f"{len(exp_df)}")
