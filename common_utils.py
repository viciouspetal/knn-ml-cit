import pandas as pd
import numpy as np

def load_data(path, columns):
    """
    Loads a given dataset with given headers

    :param path: paths to dataset
    :return: dataset with headers loaded in a pandas dataframe
    """
    df = pd.read_csv(path, names=columns, header=None)
    return df

def calculate_distances(data_points, query_instance):
    """
    Calculates a distance matrix for each of the records detailing how far each datapoint is from a given query instance.
    Additionally computes a sorted array detailing indices of the smallest to largest distance from a given
    query point, from smallest (or closest point to query instance) to largest.

    :param data_points: data points of a given dataset
    :param query_instance: instance of a dataset for which the distance matrix will be computed for
    :return:
    """

    # row wise sum with a negative lookahead - starts at the 2nd last column - ignores the last column which in
    # this instance is the actual recorded classification
    distance_matrix = np.sqrt(((data_points - query_instance) ** 2).sum(-1))

    # sorts the distance matrix and returns indices of elements from smallest distance value to largest
    sorted_distance_matrix = np.argsort(distance_matrix)[np.in1d(np.argsort(distance_matrix),np.where(distance_matrix),1)]
    return distance_matrix, sorted_distance_matrix