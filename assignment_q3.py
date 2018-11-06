import numpy as np
import common_utils as cu
import argparse


class Q3:
    path_to_folder = './dataset/regression'
    path_to_training_file = path_to_folder + '/trainingData.csv'
    path_to_test = path_to_folder + '/testData.csv'

    def main(self, path_to_dataset, k_value=5):
        data_points = cu.load_data(path_to_dataset, None)

        prediction_values = [] #initializing container for values predicted

        for index, row in data_points.iterrows():
            dist_matrix, sorted_matrix_indices = cu.calculate_distances(data_points.loc[:,0:11].values, row[0:12].values)

            prediction_values.append(self.calculate_regression(data_points, sorted_matrix_indices, k_value))

        self.calculate_r_squared(data_points[12], prediction_values)

    def calculate_r_squared(self, actual_data, predicted_data):
        """
        Calculates R\u00b2 coefficient using sum of squared residuals (SSR) and total sum of squares (TSS).
        The closer the R\u00b2 coefficient is to 1 the better the model fits the data.

        SSR is a sum of all individual residuals squared, where a residual is defined as a difference between predicted
        and actual value.
        TSS is the sum of squared difference between the average value of the feature in the dataset and the value
        predicted by regression algorithm.

        :param actual_data: target feature in the dataset
        :param predicted_data: values predicted for the given feature in the dataset
        :return: R\u00b2 coefficient between values of 0 and 1
        """

        # calculating SSR and TSS
        ssr = np.square(actual_data - predicted_data).sum()
        tss = np.square(np.average(actual_data)-predicted_data).sum()

        #print('Sum of squared residuals (SSR) is {0}'.format(ssr))
        #print('Total sum of squares (TSS) is {0}'. format(tss))

        r2_score = 1 - (ssr/tss)
        print('R\u00b2 (R squared) coefficient is {0}'. format(r2_score))
        print('Accuracy of the model is: {0} %'.format(r2_score * 100))
        return r2_score

    def calculate_regression(self, data_points, sorted_dist_indices, k_value):
        """
        Predicts a regression value for a given query instance based on its closest neighbours.

        :param data_points: points in the dataset
        :param sorted_dist_indices: indices of the sorted distance matrix defining which data points are closest to a
                given query instance
        :param k_value: value of closest neighbours to be taken into account
        :return: predicted regression value for query instance
        """
        closest_neighbours = sorted_dist_indices[:k_value]

        # each neighbour will generate it's own predicted value for the query instance.
        # Those values are then averaged out to produce actual prediction
        data_per_neighbour_to_average = []

        for i in closest_neighbours:
            data_per_neighbour_to_average.append(data_points.values[i][12])

        return np.average(data_per_neighbour_to_average)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",  help="Runs the classification on either training or test dataset. Allowed values: training, test")
    parser.add_argument("--k_value",  help="Runs the classification on either training or test dataset. Allowed values: training, test")

    args = parser.parse_args()
    subject = Q3()

    if args.run == 'training':
        subject.main(subject.path_to_training_file)
    elif args.run == 'test':
        subject.main(subject.path_to_test)
    else:
        subject.main(subject.path_to_test)