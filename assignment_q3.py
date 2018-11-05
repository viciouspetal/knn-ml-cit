import numpy as np
import common_utils as cu

class Q3:
    path_to_folder = './dataset/regression'
    path_to_training_file = path_to_folder + '/trainingData.csv'
    path_to_test = path_to_folder + '/testData.csv'

    def main(self):
        training_data = cu.load_data(self.path_to_training_file, None)
        df_test = cu.load_data(self.path_to_test, None)

        prediction_values = []

        for index, row in training_data.iterrows():
            dist_matrix, sorted_matrix_indices = cu.calculate_distances(training_data.values, row.values)

            prediction_values.append(self.calculate_regression(training_data, sorted_matrix_indices, 5))

        # epsilon = residual, defined as sum of all residuals squared. Residual = actual value - predicted value
        epsilon = np.square(training_data[12] - prediction_values).sum()
        print('RSS is {0}'.format(epsilon))

        # total sum of squares (TSS) = ((y_average - y_predicted) ^2 ).sum()
        average_of_training_data = np.average(training_data[12])

        tss = np.square(average_of_training_data-prediction_values).sum()
        print('TSS is {0}'. format(tss))

        accuracy = 1 - (epsilon/tss)
        print('R squared coefficient is {0}'. format(accuracy))
        print('Accuracy of the model is: {0}'.format(accuracy*100))


    def calculate_regression(self, data_points, sorted_dist_indices, k_value):
        closest_neighbours = sorted_dist_indices[:k_value]

        data_per_neighbour_to_average = []

        for i in closest_neighbours:
            data_per_neighbour_to_average.append(data_points.values[i][12])

        return np.average(data_per_neighbour_to_average)


if __name__ == '__main__':
    subject = Q3()
    subject.main()