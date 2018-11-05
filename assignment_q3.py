import numpy as np
import common_utils as cu

class Q3:
    path_to_folder = './dataset/regression'
    path_to_training_file = path_to_folder + '/trainingData.csv'
    path_to_test = path_to_folder + '/testData.csv'

    def main(self):
        training_data = cu.load_data(self.path_to_test, None)
        # print('Training data shape {0}'.format(data.shape))
        # print(data.describe())

        # test_data = cu.load_data(self.path_to_test, None)
        # print('Test data shape {0}'.format(test_data.shape))
        # print(data.describe())

        prediction_values = []

        for index, row in training_data.iterrows():
            dist_matrix, sorted_matrix_indices = cu.calculate_distances(training_data.values, row.values)

            prediction_values.append(self.calculate_regression(training_data, sorted_matrix_indices, 5))
        #print(training_data)
        #print(prediction_values)

        # epsilon = residual, defined as sum of all residuals squared. Residual = actual value - predicted value
        epsilon= np.square(training_data[12] - prediction_values).sum()
        print('Epsilon is {0}'.format(epsilon))

    def calculate_regression(self, data_points, sorted_dist_indices, k_value):
        closest_neighbours = sorted_dist_indices[:k_value]

        data_per_neighbour_to_average = []

        for i in closest_neighbours:
            data_per_neighbour_to_average.append(data_points.values[i][12])

        return np.average(data_per_neighbour_to_average)


if __name__ == '__main__':
    subject = Q3()
    subject.main()