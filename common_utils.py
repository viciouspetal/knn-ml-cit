import pandas as pd
import numpy as np

def load_data(path, columns):
    """
    For a given path to file it loads a dataset with given headers.

    :param path: paths to dataset
    :param columns: columns specified for a given dataset
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

    # row wise sum with a negative lookahead
    distance_matrix = euclideanDistance(data_points, query_instance)

    #print('For query instance of {0} the distance matrix is {1}'.format(query_instance,distance_matrix))

    # sorts the distance matrix and returns indices of elements from smallest distance value to largest
    sorted_distance_matrix = np.argsort(distance_matrix)[np.in1d(np.argsort(distance_matrix),np.where(distance_matrix),1)]
    return distance_matrix, sorted_distance_matrix



def euclideanDistance(data_points, query_instance):
    """
    Calculate euclidean distance

    :param data_points: data points of a given dataset
    :param query_instance: instance of a dataset for which the distance matrix will be computed for

    :return: distance value
    """
    return np.sqrt(((data_points - query_instance) ** 2).sum(-1))


def manhattanDistance(data_points, query_instance):
    """
    Calculate manhattan distance

    :param data_points: data points of a given dataset
    :param query_instance: instance of a dataset for which the distance matrix will be computed for

    :return: distance value
    """
    return np.abs(data_points - query_instance).sum(-1)


def minkowskiDistance(data_points, query_instance, p_value = 1):
    """
    Calculate minkowski distance

    :param data_points: data points of a given dataset
    :param query_instance: instance of a dataset for which the distance matrix will be computed for
    :param p_value:

    :return: distance value
    """
    return np.abs(((data_points - query_instance) / p_value) / (1 / p_value)).sum(-1)

def clean_cancer_dataset(df_training):
    """
    Checks and cleans the dataset of any potential impossible values, e.g. bi-rads columns, the 1st only allows
    values in the range of 1-5, ordinal
    Age, 2nd column, cannot be negative, integer
    Shape, 3rd column, only allows values between 1 and 4, nominal
    Margin, only allows a range of 1 to 5, nominal
    Density only allows values between 1-4,ordinal.

    All deletions will be performed in place.
    :return: cleaned up dataframe, count of removed points
    """
    rows_pre_cleaning = df_training.shape[0]

    df_training.drop(df_training.index[df_training['bi_rads'] > 5], inplace=True)
    df_training.drop(df_training.index[df_training['shape'] > 4], inplace=True)
    df_training.drop(df_training.index[df_training['margin'] > 5], inplace=True)
    df_training.drop(df_training.index[df_training['density'] > 4], inplace=True)

    rows_removed = rows_pre_cleaning - df_training.shape[0]
    return df_training, rows_removed

def compute_classification_accuracy(correctly_classified, incorrectly_classified):
    """
    Computes the accuracy of the model based on the number of correctly and incorrectly classified points.
    Expresses accuracy as a percentage value.

    :param correctly_classified: count of correctly classified data points
    :param incorrectly_classified: count of incorrectly classified data points
    :return: accuracy score
    """
    accuracy = (correctly_classified / (correctly_classified + incorrectly_classified)) * 100
    return accuracy
