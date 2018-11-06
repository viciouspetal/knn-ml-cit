import numpy as np
import common_utils as cu
import argparse

class Q2:
    path_to_cancer = './dataset/cancer'
    path_to_cancer_training = path_to_cancer + '/trainingData2.csv'
    path_to_test=path_to_cancer+'/testData2.csv'
    cancer_dataset_column_headers = ['bi_rads', 'age', 'shape', 'margin', 'density', 'severity']

    def classify_points_with_weight(self, dist, sorted, dataset, k_value):
        """
        Calculate the classification based on weighted distance. The largest weight will be used to classify the
        query instance.
        """
        classes = {}
        weights = {}

        closest_neighbours = sorted[:k_value]

        # Looping through the k-closest values. Each result is stored in a new dictionary object to enable quick lookup.
        for i in closest_neighbours:
            key = dataset[i][5]

            if not key in classes:
                classes[key] = [dist[i]]
            else:
                classes[key].append(dist[i])

        # Looping through above dictionary and calculating the total weight for each classification.
        for classification in classes:
            total = 0
            for i in classes[classification]:
                total += (1 / i)

            weights[classification] = total

        largest_weight = 0
        chosen_classification = None

        # Looping through the weights dictionary in order to locate the key  or classification with the largest weight
        # This will become the classification of the query instance object
        for classification in weights:
            if weights[classification] > largest_weight:
                largest_weight = weights[classification]
                chosen_classification = classification

        return chosen_classification

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

    def main(self, path_to_data=path_to_cancer_training, headers= cancer_dataset_column_headers, k_value=3, alg_to_use='euclidean', p=1):
        data_points = cu.load_data(path_to_data, headers)
        df_training, row_count_removed = cu.clean_cancer_dataset(data_points)

        print('The dataset has been cleaned of the impossible values. {0} rows have been removed'.format(row_count_removed))

        correctly_classified = 0
        incorrectly_classified = 0

        for index, row in data_points.iterrows():
            dist_matrix, sorted_matrix_indices = self.calculate_distances(data_points.loc[:,'bi_rads':'density'].values, row[0:5].values, alg_to_use, p)

            classification = self.classify_points_with_weight(dist_matrix, sorted_matrix_indices, data_points.values, k_value)

            if classification == row.values[5]:
                correctly_classified += 1
            else:
                incorrectly_classified += 1

        accuracy = cu.compute_classification_accuracy(correctly_classified, incorrectly_classified)

        print('For the k = {0} using {1} distance weighing algorithm, the accuracy is: {2} %,'.format(k_value, alg_to_use, accuracy))
        #print('{0}, {1}, {2}'.format(k_value, alg_to_use, accuracy))

    def best_params(self, path_to_data):
        k_target = 10
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

    args = parser.parse_args()
    subject = Q2()
    if args.run == 'training':
        subject.main(subject.path_to_cancer_training)
    elif args.run == 'test':
        subject.main(subject.path_to_test)
    elif args.run == 'best':
        #subject.best_params(subject.path_to_test)
        subject.best_params(subject.path_to_cancer_training)
    elif args.run == 'mink':
        subject.best_mink(subject.path_to_test, 10)
    else:
        subject.main(subject.path_to_test)
