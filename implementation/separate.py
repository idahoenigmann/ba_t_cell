import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def import_all_data():
    all_data = {"human_positive": None, "mouse_positive": None, "mouse_negative": None}
    all_indices = {"human_positive": None, "mouse_positive": None, "mouse_negative": None}
    for file in all_data.keys():
        all_data[file] = np.loadtxt(f'intermediate/particle_parameters_{file}.csv', delimiter=',')
        with open(f'intermediate/particle_parameters_{file}.csv', 'r') as f_in:
            header = f_in.readline()
            header = header.translate({ord(c): None for c in '# \n'})
            all_indices[file] = header.split(',')
    return all_data, all_indices


def index_term_to_data(data, file, term):
    def term_sub_helper(data, file, term):
        token_lst = term.split("-")
        if len(token_lst) == 1:
            return data[file][:, indices[file].index(term)]
        else:
            tmp = data[file][:, indices[file].index(token_lst[0])]
            for i in range(1, len(token_lst)):
                tmp -= data[file][:, indices[file].index(token_lst[i])]
            return tmp

    token_lst = term.split("+")
    if len(token_lst) == 1:
        return term_sub_helper(data, file, term)
    else:
        tmp = term_sub_helper(data, file, token_lst[0])
        for i in range(1, len(token_lst)):
            tmp += term_sub_helper(data, file, token_lst[i])
        return tmp


def visualize_2d_compare(data, x_axis, y_axis):
    for f in data.keys():
        plt.scatter(index_term_to_data(data, f, x_axis), index_term_to_data(data, f, y_axis))

    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.show()


def visualize_3d_compare(data, x_axis, y_axis, z_axis):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for f in data.keys():
        ax.scatter(index_term_to_data(data, f, x_axis), index_term_to_data(data, f, y_axis),
                   index_term_to_data(data, f, z_axis))

    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_zlabel(z_axis)
    plt.show()


if __name__ == "__main__":
    """
    plot multiple datasets in single plot with various axis
    """

    data, indices = import_all_data()

    dim = 2

    tmp = ["u", "a", "d", "k", "w-start", "t-start"]
    if dim == 2:
        for x in range(len(tmp)):
            for y in range(x + 1, len(tmp)):
                visualize_2d_compare(data, tmp[x], tmp[y])
    elif dim == 3:
        for x in range(len(tmp)):
            for y in range(x + 1, len(tmp)):
                for z in range(y + 1, len(tmp)):
                    visualize_3d_compare(data, tmp[x], tmp[y], tmp[z])
