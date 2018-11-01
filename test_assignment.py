import unittest
import pandas as pd
import numpy as np

from assignment import Assignment


class TestAssignment(unittest.TestCase):
    trainingDf = None
    testDf = None

    @classmethod
    def setUpClass(cls):
        TestAssignment.trainingDf = pd.read_csv(Assignment.path_to_cancer_training, names=Assignment.cancer_dataset_column_headers, header=None)

    def test_calculate_distance_when_end_larger_than_start_then_distance_is_calculated(self):
        under_test = Assignment()
        result = under_test.calculate_distances(TestAssignment.trainingDf.values, TestAssignment.trainingDf.values)

        print('resulting array is {0}'. format(result))

        result_elements_that_are_ot_zero = np.where(~result.any(axis=0))[0]
        self.assertEqual(result_elements_that_are_ot_zero[0], 0)
        self.assertEqual(len(result), 514)

    def test_load_data_when_loading_training_data_then_the_count_is_correct(self):
        expected_count = 514
        under_test = Assignment()
        result = under_test.load_data(under_test.path_to_cancer_training)
        self.assertEqual(expected_count, len(result))


if __name__ == '__main__':
    unittest.main()
