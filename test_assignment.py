import unittest
from assignment import Assignment


class TestAssignment(unittest.TestCase):

    def test_calculate_distance_when_end_larger_than_start_then_distance_is_calculated(self):
        under_test = Assignment()
        result = under_test.calculate_distances(6, 10)
        self.assertEqual(result, 4)

    def test_calculate_distance_when_start_larger_than_end_then_distance_is_calculated(self):
        under_test = Assignment()
        result = under_test.calculate_distances(10, 6)
        self.assertEqual(result, 4)

    def test_load_data_when_loading_training_data_then_the_count_is_correct(self):
        expected_count = 514
        under_test = Assignment()
        result = under_test.load_data(under_test.path_to_cancer_training)
        self.assertEqual(expected_count, len(result))


if __name__ == '__main__':
    unittest.main()
