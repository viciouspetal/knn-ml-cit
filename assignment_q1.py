import numpy as np
import operator
import common_utils as cu
import argparse


class Q1:
    path_to_cancer = './dataset/cancer'
    path_to_cancer_training = path_to_cancer + '/trainingData2.csv'
    path_to_test=path_to_cancer+'/testData2.csv'
    cancer_dataset_column_headers = ['bi_rads', 'age', 'shape', 'margin', 'density', 'severity']

    def calculate_distances(self, data_points, query_instance):
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

    def main(self, path=path_to_cancer_training, k_value=3):
        """
        Directs the analysis process by orchestrating calls to relevant pieces of the KNN algorithm implementation

        :param path: path to file to be analyzed. Default value is assumed to correspond to trainingData2.csv
        in cancer directory
        """
        # first need to load the training dataset
        df_training = cu.load_data(path, self.cancer_dataset_column_headers)
        df_training, row_count_removed = cu.clean_cancer_dataset(df_training)

        print('The dataset has been cleaned of the impossible values. {0} rows have been removed'.format(row_count_removed))

        correctly_classified = 0
        incorrectly_classified = 0

        # passing pandas dataframe converted into a numpy array as well as each query instance in the dataset to calculate distance matrix
        for index, row in df_training.iterrows():
            dist_matrix, sorted_matrix_indices = cu.calculate_distances(df_training.loc[:,'bi_rads':'density'].values, row[0:5].values)

            classification = self.classify_points(df_training, sorted_matrix_indices, k_value)

            if classification == row.values[5]:
                correctly_classified += 1
            else:
                incorrectly_classified += 1

        accuracy = cu.compute_classification_accuracy(correctly_classified, incorrectly_classified)

        print('Accuracy is: ', accuracy, '%')

    def classify_points(self, dataset, sorted_dist_array, k_value):
        """
        Classifies each data point according to its nearest neighbours using a vote system.
        Verifies the findings using the last column, the severity, as a measure of accuracy.


        :param dataset: distance matrix
        :param sorted_dist_array: array of indices of distance element in the dataset,
        from closest to query instance to the furthest
        :param k_value: value of the nearest neighbours to be taken into account classifying a query instance
        :return:
        """

        votes = {}
        closest_neighbours = sorted_dist_array[:k_value]

        for i in closest_neighbours:
            key = dataset.values[i][5]
            if not key in votes:
                votes[key] = 1
            else:
                votes[key] += 1

        return max(votes.items(), key=operator.itemgetter(1))[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",  help="Runs the classification on either training or test dataset. Allowed values: training, test")

    args = parser.parse_args()
    subject = Q1()
    if args.run == 'training':
        subject.main(subject.path_to_cancer_training)
    elif args.run == 'test':
        subject.main(subject.path_to_test)
    else:
        subject.main(subject.path_to_test)


