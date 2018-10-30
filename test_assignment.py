import unittest
from assignment import *


class Test_Assignment(unittest.TestCase):

    def test_calculate_distance_when_end_larger_than_start_then_distance_is_calculated(self):
        under_test = assignment()
        result = under_test.calculate_distance(6, 10)
        print('test value is {0}'.format(result))
        self.assertEquals(result, 4)


if __name__ == '__main__':
    unittest.main()
