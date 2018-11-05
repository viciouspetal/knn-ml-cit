import unittest
import pandas as pd
import numpy as np

from assignment import Assignment


class TestAssignment(unittest.TestCase):
    trainingDf = None

    @classmethod
    def setUpClass(cls):
        TestAssignment.trainingDf = pd.read_csv(Assignment.path_to_cancer_training, names=Assignment.cancer_dataset_column_headers, header=None)

    def test_load_data_when_loading_training_data_then_the_count_is_correct(self):
        expected_count = 514
        under_test = Assignment()
        result = under_test.load_data(under_test.path_to_cancer_training)
        self.assertEqual(expected_count, len(result))

    def test_clean_dataset_when_invalid_records_present_then_they_are_removed(self):
        testDf = pd.read_csv('./test_dataset/dataset_with_invalid_records.csv', names=Assignment.cancer_dataset_column_headers, header=None)

        record_count_before_clean = testDf.shape[0]
        under_test = Assignment()
        testDf, removed_count = under_test.clean_dataset(testDf)

        self.assertEqual(record_count_before_clean, 20)
        self.assertEqual(removed_count, 4)
        self.assertEqual(testDf.shape[0], 16)

    def test_compute_accuracy_when_correct_and_incorrect_counts_are_provided_then_accuracy_is_computed(self):
        under_test = Assignment()

        self.assertEqual(75, under_test.compute_accuracy(75, 25))

    def test_calculate_distance_when_1_query_instance_provided_then_distance_matrix_is_calculated_and_sorted(self):
        under_test = Assignment()
        result, sorted_result_indices = under_test.calculate_distances(TestAssignment.trainingDf.values, TestAssignment.trainingDf.values[0])

        self.assertIsNotNone(result)
        self.assertIsNotNone(sorted_result_indices)
        self.assertEqual(514, len(result))
        self.assertEqual(513, len(sorted_result_indices))

    def test_calculate_distance_when_same_data_is_fed_twice_then_resulting_distances_are_zeroed_out(self):
        under_test = Assignment()
        result, sorted = under_test.calculate_distances(TestAssignment.trainingDf.values, TestAssignment.trainingDf.values)

        #print('resulting array is {0}'. format(result))

        result_elements_that_are_ot_zero = np.where(~result.any(axis=0))[0]
        self.assertEqual(result_elements_that_are_ot_zero[0], 0)
        self.assertEqual(len(result), 514)


if __name__ == '__main__':
    unittest.main()
