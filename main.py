import pandas as pd


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
    return 0


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


if __name__ == '__main__':
    data = read_data()
