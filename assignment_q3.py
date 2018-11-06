import numpy as np
import common_utils as cu
import argparse


class Q3:
    path_to_folder = './dataset/regression'
    path_to_training_file = path_to_folder + '/trainingData.csv'
    path_to_test = path_to_folder + '/testData.csv'

    def main(self, path_to_dataset, k_value=5, alg_to_use='euclidean', p=1):
        data_points = cu.load_data(path_to_dataset, None)

        prediction_values = [] #initializing container for values predicted

        for index, row in data_points.iterrows():
            dist_matrix, sorted_matrix_indices = self.calculate_distances(data_points.loc[:,0:11].values, row[0:12].values, alg_to_use, p)

            prediction_values.append(self.calculate_regression(data_points, sorted_matrix_indices, k_value))

        r2_score=self.calculate_r_squared(data_points[12], prediction_values)
        #print('{0}, {1}, {2}'.format(k_value, alg_to_use, r2_score))
        print('R\u00b2 (R squared) coefficient is {0}'. format(r2_score))
        print('Accuracy of the model is: {0} %'.format(r2_score * 100))

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

    def calculate_distances(self, data_points, query_instance, alg_to_use='euclidean', p=1):
        """
        Calculates the distance based on selected algorithm. The supported values are: euclidean, manhattan and minkowski
        """

        if alg_to_use == 'euclidean' or alg_to_use == 1 :
            dist = cu.euclideanDistance(data_points, query_instance)
        elif alg_to_use == 'manhattan' or alg_to_use == 2:
            dist = cu.manhattanDistance(data_points, query_instance)
        elif alg_to_use == 'minkowski' or alg_to_use == 3:
            dist = cu.minkowskiDistance(data_points, query_instance, p)

        # Sort the indexes of distances acquired form closest to furthest
        sorted_indices = np.argsort(dist)[np.in1d(np.argsort(dist), np.where(dist), 1)]

        return dist, sorted_indices

    def best_params(self, path_to_data, k_target=10):
        distance_algs = ['euclidean', 'manhattan', 'minkowski']

        for alg in distance_algs:
            for k in range(1, k_target+1):
                self.main(path_to_data, k_value=k, alg_to_use=alg)

    def best_mink(self, path_to_data, k_target = 10, p_target=10):
        for k in range(1, k_target+1):
            for p in range(1, p_target+1):
                self.main(path_to_data, k_value=k, alg_to_use='minkowski', p=p)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",  help="Runs the classification on either training or test dataset. Allowed values: training, test")
    parser.add_argument("--k_value",  help="Runs the classification on either training or test dataset. Allowed values: training, test")

    args = parser.parse_args()
    subject = Q3()

    if args.run == 'training':
        subject.main(subject.path_to_training_file)
    elif args.run == 'test':
        subject.main(subject.path_to_test, 4)
    elif args.run == 'best':
        #subject.best_params(subject.path_to_test)
        subject.best_params(subject.path_to_test, 10)
    elif args.run == 'mink':
        subject.best_mink(subject.path_to_test, 10)
    else:
        subject.main(subject.path_to_test)