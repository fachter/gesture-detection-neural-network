import unittest

import numpy as np

from ..neural_net.data_scaler import DataScaler


class MyTestCase(unittest.TestCase):
    input_data = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [2, 4, 6, 8]
    ])

    def test_given_same_values_for_each_row(self):
        expected_scaled_data = np.array([
            [2, 4, 6, 8],
            [10, 12, 14, 16],
            [4, 8, 12, 16]
        ])

        # should scale 0.5, because x_diff = 0.4 & y_diff = 0.3 ==> math.sqrt(0.3**2 + 0.4++2) == 0.5
        scaled_data = DataScaler().scale_data(
            left_shoulder_x=np.array([0.3, 0.3, 0.3]),
            left_shoulder_y=np.array([0.2, 0.2, 0.2]),
            right_shoulder_x=np.array([0.7, 0.7, 0.7]),
            right_shoulder_y=np.array([0.5, 0.5, 0.5]),
            data_to_scale=self.input_data
        )

        np.testing.assert_array_almost_equal(scaled_data, expected_scaled_data)

    def test_given_different_values_for_each_row(self):
        self.input_data = np.array([
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
        ])
        expected_scaled_data = np.array([
            [2, 4, 6, 8],
            [1_000_000, 2_000_000, 3_000_000, 4_000_000],
            [10, 20, 30, 40]
        ])

        # should scale 0.5, because x_diff = 0.4 & y_diff = 0.3 ==> math.sqrt(0.3**2 + 0.4++2) == 0.5
        scaled_data = DataScaler().scale_data(
            left_shoulder_x=np.array([0.3, 0.3, 0.5]),
            left_shoulder_y=np.array([0.2, 0.4, 0.4]),
            right_shoulder_x=np.array([0.7, 0.3, 0.6]),
            right_shoulder_y=np.array([0.5, 0.4, 0.4]),
            data_to_scale=self.input_data
        )

        np.testing.assert_array_almost_equal(scaled_data, expected_scaled_data)


if __name__ == '__main__':
    unittest.main()
