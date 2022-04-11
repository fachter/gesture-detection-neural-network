import unittest
import numpy as np
import pandas as pd
from ..neural_net.data_preprocessor import DataPreprocessor
from unittest.mock import MagicMock
from ..neural_net.data_scaler import DataScaler


class MyTestCase(unittest.TestCase):
    scaler_mock = DataScaler()
    data_processor = DataPreprocessor(data_scaler=scaler_mock)
    first_row = [
        # left_pinky_x
        0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
        0.01, 0.01, 0.01,
        # left_pinky_y
        0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
        0.01, 0.01, 0.01,
        # right_pinky_x
        0.0002, 0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0064, 0.0128, 0.0256, 0.0512, 0.1024,
        -0.1024, -0.0512, -0.0256, -0.0128, -0.0064, -0.0032, -0.0016, -0.0008, -0.0004,
        # right_pinky_y
        0.0002, 0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0064, 0.0128, 0.0256, 0.0512, 0.1024,
        -0.1024, -0.0512, -0.0256, -0.0128, -0.0064, -0.0032, -0.0016, -0.0008, -0.0004,
        # left_index_x
        0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
        0.01, 0.01, 0.01,
        # left_index_y
        0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03,
        0.03, 0.03, 0.03,
        # right_index_x
        0.0002, 0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0064, 0.0128, 0.0256, 0.0512, 0.1024,
        -0.1024, -0.0512, -0.0256, -0.0128, -0.0064, -0.0032, -0.0016, -0.0008, -0.0004,
        # right_index_y
        0.0002, 0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0064, 0.0128, 0.0256, 0.0512, 0.1024,
        -0.1024, -0.0512, -0.0256, -0.0128, -0.0064, -0.0032, -0.0016, -0.0008, -0.0004,
        # left_thumb_x
        0.0002, 0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0064, 0.0128, 0.0256, 0.0512, 0.1024,
        -0.1024, -0.0512, -0.0256, -0.0128, -0.0064, -0.0032, -0.0016, -0.0008, -0.0004,
        # left_thumb_y
        0.0002, 0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0064, 0.0128, 0.0256, 0.0512, 0.1024,
        -0.1024, -0.0512, -0.0256, -0.0128, -0.0064, -0.0032, -0.0016, -0.0008, -0.0004,
        # right_thumb_x
        0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
        0.01, 0.01, 0.01,
        # right_thumb_y
        0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
        0.01, 0.01, 0.01,
        # left_elbow_x
        0.0002, 0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0064, 0.0128, 0.0256, 0.0512, 0.1024,
        -0.1024, -0.0512, -0.0256, -0.0128, -0.0064, -0.0032, -0.0016, -0.0008, -0.0004,
        # left_elbow_y
        0.0002, 0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0064, 0.0128, 0.0256, 0.0512, 0.1024,
        -0.1024, -0.0512, -0.0256, -0.0128, -0.0064, -0.0032, -0.0016, -0.0008, -0.0004,
        # right_elbow_x
        0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
        0.01, 0.01, 0.01,
        # right_elbow_y
        0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03,
        0.03, 0.03, 0.03,
        # left_shoulder_x
        0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
        0.01, 0.01, 0.01,
        # left_shoulder_y
        0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03,
        0.03, 0.03, 0.03,
        # right_shoulder_x
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0,
        # right_shoulder_y
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0,

        # distance_thumb_right_mouth_right_x
        0.49, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41, 0.40, 0.39, 0.38, 0.37, 0.36, 0.35, 0.34, 0.33,
        0.32, 0.31, 0.30,
        # distance_thumb_right_mouth_right_y
        0.49, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41, 0.40, 0.39, 0.38, 0.37, 0.36, 0.35, 0.34, 0.33,
        0.32, 0.31, 0.30,
        # distance_thumb_left_mouth_left_x
        0.5 - 0.3, 0.5 - 0.3002, 0.5 - 0.3004, 0.5 - 0.3008, 0.5 - 0.3016, 0.5 - 0.3032, 0.5 - 0.3064,
        0.5 - 0.3128, 0.5 - 0.3256, 0.5 - 0.3512, 0.5 - 0.4024, 0.5 - 0.5048, 0.5 - 0.4024, 0.5 - 0.3512,
        0.5 - 0.3256, 0.5 - 0.3128, 0.5 - 0.3064, 0.5 - 0.3032, 0.5 - 0.3016, 0.5 - 0.3008,

        # distance_thumb_left_mouth_left_y
        0.5 - 0.3, 0.5 - 0.3002, 0.5 - 0.3004, 0.5 - 0.3008, 0.5 - 0.3016, 0.5 - 0.3032, 0.5 - 0.3064,
        0.5 - 0.3128, 0.5 - 0.3256, 0.5 - 0.3512, 0.5 - 0.4024, 0.5 - 0.5048, 0.5 - 0.4024, 0.5 - 0.3512,
        0.5 - 0.3256, 0.5 - 0.3128, 0.5 - 0.3064, 0.5 - 0.3032, 0.5 - 0.3016, 0.5 - 0.3008,
    ]

    def test_given_data_of_at_least_20_frames_then_preprocess_relative_data(self):
        self.scaler_mock.get_scale_value = MagicMock(
            return_value=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        frames = pd.read_csv("test_processing.csv")
        # data_processor = DataPreprocessor(data_scaler=self.scaler_mock)
        expected_result = np.array([
            self.first_row
        ])

        result = self.data_processor.preprocess_data(frames[:21])

        np.testing.assert_array_almost_equal(result[:, :40], expected_result[:, 0:40])
        np.testing.assert_array_almost_equal(result[:, 40:80], expected_result[:, 40:80])
        np.testing.assert_array_almost_equal(result[:, 80:120], expected_result[:, 80:120])
        np.testing.assert_array_almost_equal(result[:, 120:160], expected_result[:, 120:160])
        np.testing.assert_array_almost_equal(result[:, 160:200], expected_result[:, 160:200])
        np.testing.assert_array_almost_equal(result[:, 200:240], expected_result[:, 200:240])
        np.testing.assert_array_almost_equal(result[:, 240:], expected_result[:, 240:])
        np.testing.assert_array_almost_equal(result, expected_result, decimal=10)

    def test_given_more_than_20_frames_then_return_multiple_values(self):
        self.scaler_mock.get_scale_value = MagicMock(
            return_value=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        frames = pd.read_csv("test_processing.csv")
        expected_result = np.array([
            self.first_row,
            [
                # left_pinky_x
                0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                0.01, 0.01, 0.01,
                # left_pinky_y
                0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                0.01, 0.01, 0.01,
                # right_pinky_x
                0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0064, 0.0128, 0.0256, 0.0512, 0.1024,
                -0.1024, -0.0512, -0.0256, -0.0128, -0.0064, -0.0032, -0.0016, -0.0008, -0.0004, -0.0002,
                # right_pinky_y
                0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0064, 0.0128, 0.0256, 0.0512, 0.1024,
                -0.1024, -0.0512, -0.0256, -0.0128, -0.0064, -0.0032, -0.0016, -0.0008, -0.0004, -0.0002,
                # left_index_x
                0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                0.01, 0.01, 0.01,
                # left_index_y
                0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03,
                0.03, 0.03, 0.03,
                # right_index_x
                0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0064, 0.0128, 0.0256, 0.0512, 0.1024,
                -0.1024, -0.0512, -0.0256, -0.0128, -0.0064, -0.0032, -0.0016, -0.0008, -0.0004, -0.0002,
                # right_index_y
                0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0064, 0.0128, 0.0256, 0.0512, 0.1024,
                -0.1024, -0.0512, -0.0256, -0.0128, -0.0064, -0.0032, -0.0016, -0.0008, -0.0004, -0.0002,
                # left_thumb_x
                0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0064, 0.0128, 0.0256, 0.0512, 0.1024,
                -0.1024, -0.0512, -0.0256, -0.0128, -0.0064, -0.0032, -0.0016, -0.0008, -0.0004, -0.0002,
                # left_thumb_y
                0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0064, 0.0128, 0.0256, 0.0512, 0.1024,
                -0.1024, -0.0512, -0.0256, -0.0128, -0.0064, -0.0032, -0.0016, -0.0008, -0.0004, -0.0002,
                # right_thumb_x
                0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                0.01, 0.01, 0.01,
                # right_thumb_y
                0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                0.01, 0.01, 0.01,
                # left_elbow_x
                0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0064, 0.0128, 0.0256, 0.0512, 0.1024,
                -0.1024, -0.0512, -0.0256, -0.0128, -0.0064, -0.0032, -0.0016, -0.0008, -0.0004, -0.0002,
                # left_elbow_y
                0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0064, 0.0128, 0.0256, 0.0512, 0.1024,
                -0.1024, -0.0512, -0.0256, -0.0128, -0.0064, -0.0032, -0.0016, -0.0008, -0.0004, -0.0002,
                # right_elbow_x
                0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                0.01, 0.01, 0.01,
                # right_elbow_y
                0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03,
                0.03, 0.03, 0.03,
                # left_shoulder_x
                0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                0.01, 0.01, 0.01,
                # left_shoulder_y
                0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03,
                0.03, 0.03, 0.03,
                # right_shoulder_x
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0,
                # right_shoulder_y
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0,

                # distance_thumb_right_mouth_right_x
                0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41, 0.40, 0.39, 0.38, 0.37, 0.36, 0.35, 0.34, 0.33,
                0.32, 0.31, 0.30, 0.29,
                # distance_thumb_right_mouth_right_y
                0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41, 0.40, 0.39, 0.38, 0.37, 0.36, 0.35, 0.34, 0.33,
                0.32, 0.31, 0.30, 0.29,
                # distance_thumb_left_mouth_left_x
                0.5 - 0.3002, 0.5 - 0.3004, 0.5 - 0.3008, 0.5 - 0.3016, 0.5 - 0.3032, 0.5 - 0.3064, 0.5 - 0.3128,
                0.5 - 0.3256, 0.5 - 0.3512, 0.5 - 0.4024, 0.5 - 0.5048, 0.5 - 0.4024, 0.5 - 0.3512,
                0.5 - 0.3256, 0.5 - 0.3128, 0.5 - 0.3064, 0.5 - 0.3032, 0.5 - 0.3016, 0.5 - 0.3008, 0.5 - 0.3004,

                # distance_thumb_left_mouth_left_y
                0.5 - 0.3002, 0.5 - 0.3004, 0.5 - 0.3008, 0.5 - 0.3016, 0.5 - 0.3032, 0.5 - 0.3064,
                0.5 - 0.3128, 0.5 - 0.3256, 0.5 - 0.3512, 0.5 - 0.4024, 0.5 - 0.5048, 0.5 - 0.4024, 0.5 - 0.3512,
                0.5 - 0.3256, 0.5 - 0.3128, 0.5 - 0.3064, 0.5 - 0.3032, 0.5 - 0.3016, 0.5 - 0.3008, 0.5 - 0.3004,
            ]
        ])

        result = self.data_processor.preprocess_data(frames[:22])

        np.testing.assert_array_almost_equal(result[0, :], expected_result[0, :])
        np.testing.assert_array_almost_equal(result[1, :40], expected_result[1, :40])
        np.testing.assert_array_almost_equal(result[1, 40:80], expected_result[1, 40:80])
        np.testing.assert_array_almost_equal(result[1, 80:120], expected_result[1, 80:120])
        np.testing.assert_array_almost_equal(result[1, 120:160], expected_result[1, 120:160])
        np.testing.assert_array_almost_equal(result[1, 160:200], expected_result[1, 160:200])
        np.testing.assert_array_almost_equal(result[1, 200:240], expected_result[1, 200:240])
        np.testing.assert_array_almost_equal(result[1, 240:280], expected_result[1, 240:280])
        np.testing.assert_array_almost_equal(result[1, 280:], expected_result[1, 280:])

    def test_given_ground_truth_as_well_then_return_both(self):
        frames = pd.read_csv("test_processing.csv")
        expected_result = np.array(["rotate"])
        data_processor = DataPreprocessor(data_scaler=self.scaler_mock)

        _, result = data_processor.preprocess_data(frames[:21], including_ground_truth=True)

        np.testing.assert_array_equal(result, expected_result)

    def test_given_percentage_majority_then_label_less_as_gesture(self):
        frames = pd.read_csv("test_processing.csv")[:21]
        labels = ["idle" for _ in range(8)]
        labels.extend(["rotate" for _ in range(13)])
        print(labels)
        frames["ground_truth"] = labels
        preprocessor = DataPreprocessor(data_scaler=self.scaler_mock, percentage_majority=.75)

        _, gt = preprocessor.preprocess_data(frames, including_ground_truth=True)

        self.assertEqual("idle", gt[0])

    def test_given_small_percentage_majority_then_label_more_as_gesture(self):
        frames = pd.read_csv("test_processing.csv")[:21]
        labels = ["idle" for _ in range(15)]
        labels.extend(["rotate" for _ in range(6)])
        print(labels)
        frames["ground_truth"] = labels
        preprocessor = DataPreprocessor(data_scaler=self.scaler_mock, percentage_majority=.25)

        _, gt = preprocessor.preprocess_data(frames, including_ground_truth=True)

        self.assertEqual("idle", gt[0])


    def test_given_scaler_will_return_different_than_one(self):
        self.scaler_mock.get_scale_value = MagicMock(
            return_value=[10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])

        frames = pd.read_csv("test_processing.csv")
        indices_without_scaling = (4 * 20)
        first_row_scaled: np.ndarray = np.multiply(self.first_row[:-indices_without_scaling], 10)
        expected_result = np.array([(np.append(first_row_scaled, self.first_row[-indices_without_scaling:]))])

        result = self.data_processor.preprocess_data(frames[:21])

        np.testing.assert_array_almost_equal(result[:, :40], expected_result[:, 0:40])
        np.testing.assert_array_almost_equal(result[:, 40:80], expected_result[:, 40:80])
        np.testing.assert_array_almost_equal(result[:, 80:120], expected_result[:, 80:120])
        np.testing.assert_array_almost_equal(result[:, 120:160], expected_result[:, 120:160])
        np.testing.assert_array_almost_equal(result[:, 160:200], expected_result[:, 160:200])
        np.testing.assert_array_almost_equal(result[:, 200:240], expected_result[:, 200:240])
        np.testing.assert_array_almost_equal(result[:, 240:], expected_result[:, 240:])
        np.testing.assert_array_almost_equal(result, expected_result, decimal=10)
        np.testing.assert_array_almost_equal(result[0, :], expected_result[0, :])

    def test_given_scaler_will_return_different_values_for_one_iteration(self):
        self.scaler_mock.get_scale_value = MagicMock(
            return_value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        frames = pd.read_csv("test_processing.csv")
        indices_without_scaling = (4 * 20)
        self.first_row[0:-indices_without_scaling:20] = np.multiply(self.first_row[0:-indices_without_scaling:20], 1)
        self.first_row[1:-indices_without_scaling:20] = np.multiply(self.first_row[1:-indices_without_scaling:20], 2)
        self.first_row[2:-indices_without_scaling:20] = np.multiply(self.first_row[2:-indices_without_scaling:20], 3)
        self.first_row[3:-indices_without_scaling:20] = np.multiply(self.first_row[3:-indices_without_scaling:20], 4)
        self.first_row[4:-indices_without_scaling:20] = np.multiply(self.first_row[4:-indices_without_scaling:20], 5)
        self.first_row[5:-indices_without_scaling:20] = np.multiply(self.first_row[5:-indices_without_scaling:20], 6)
        self.first_row[6:-indices_without_scaling:20] = np.multiply(self.first_row[6:-indices_without_scaling:20], 7)
        self.first_row[7:-indices_without_scaling:20] = np.multiply(self.first_row[7:-indices_without_scaling:20], 8)
        self.first_row[8:-indices_without_scaling:20] = np.multiply(self.first_row[8:-indices_without_scaling:20], 9)
        self.first_row[9:-indices_without_scaling:20] = np.multiply(self.first_row[9:-indices_without_scaling:20], 10)
        self.first_row[10:-indices_without_scaling:20] = np.multiply(self.first_row[10:-indices_without_scaling:20], 1)
        self.first_row[11:-indices_without_scaling:20] = np.multiply(self.first_row[11:-indices_without_scaling:20], 2)
        self.first_row[12:-indices_without_scaling:20] = np.multiply(self.first_row[12:-indices_without_scaling:20], 3)
        self.first_row[13:-indices_without_scaling:20] = np.multiply(self.first_row[13:-indices_without_scaling:20], 4)
        self.first_row[14:-indices_without_scaling:20] = np.multiply(self.first_row[14:-indices_without_scaling:20], 5)
        self.first_row[15:-indices_without_scaling:20] = np.multiply(self.first_row[15:-indices_without_scaling:20], 6)
        self.first_row[16:-indices_without_scaling:20] = np.multiply(self.first_row[16:-indices_without_scaling:20], 7)
        self.first_row[17:-indices_without_scaling:20] = np.multiply(self.first_row[17:-indices_without_scaling:20], 8)
        self.first_row[18:-indices_without_scaling:20] = np.multiply(self.first_row[18:-indices_without_scaling:20], 9)
        self.first_row[19:-indices_without_scaling:20] = np.multiply(self.first_row[19:-indices_without_scaling:20], 10)
        expected_result = np.array([self.first_row])

        result = self.data_processor.preprocess_data(frames[:21])

        np.testing.assert_array_almost_equal(result[:, :40], expected_result[:, 0:40])
        np.testing.assert_array_almost_equal(result[:, 40:80], expected_result[:, 40:80])
        np.testing.assert_array_almost_equal(result[:, 80:120], expected_result[:, 80:120])
        np.testing.assert_array_almost_equal(result[:, 120:160], expected_result[:, 120:160])
        np.testing.assert_array_almost_equal(result[:, 160:200], expected_result[:, 160:200])
        np.testing.assert_array_almost_equal(result[:, 200:240], expected_result[:, 200:240])
        np.testing.assert_array_almost_equal(result[:, 240:], expected_result[:, 240:])
        np.testing.assert_array_almost_equal(result, expected_result, decimal=10)
        np.testing.assert_array_almost_equal(result[0, :], expected_result[0, :])

    def test_given_relative_to_first_frame(self):
        first_row = [
            # left_pinky_x
            0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17,
            0.18, 0.19, 0.20,
            # left_pinky_y
            0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17,
            0.18, 0.19, 0.20,
            # right_pinky_x
            0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0064, 0.0128, 0.0256, 0.0512, 0.1024, 0.2048, 0.1024, 0.0512,
            0.0256, 0.0128, 0.0064, 0.0032, 0.0016, 0.0008, 0.0004,
            # right_pinky_y
            0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0064, 0.0128, 0.0256, 0.0512, 0.1024, 0.2048, 0.1024, 0.0512,
            0.0256, 0.0128, 0.0064, 0.0032, 0.0016, 0.0008, 0.0004,
            # left_index_x
            0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17,
            0.18, 0.19, 0.20,
            # left_index_y
            0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.30, 0.33, 0.36, 0.39, 0.42, 0.45, 0.48, 0.51,
            0.54, 0.57, 0.60,
            # right_index_x
            0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0064, 0.0128, 0.0256, 0.0512, 0.1024, 0.2048, 0.1024, 0.0512,
            0.0256, 0.0128, 0.0064, 0.0032, 0.0016, 0.0008, 0.0004,
            # right_index_y
            0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0064, 0.0128, 0.0256, 0.0512, 0.1024, 0.2048, 0.1024, 0.0512,
            0.0256, 0.0128, 0.0064, 0.0032, 0.0016, 0.0008, 0.0004,
            # left_thumb_x
            0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0064, 0.0128, 0.0256, 0.0512, 0.1024, 0.2048, 0.1024, 0.0512,
            0.0256, 0.0128, 0.0064, 0.0032, 0.0016, 0.0008, 0.0004,
            # left_thumb_y
            0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0064, 0.0128, 0.0256, 0.0512, 0.1024, 0.2048, 0.1024, 0.0512,
            0.0256, 0.0128, 0.0064, 0.0032, 0.0016, 0.0008, 0.0004,
            # right_thumb_x
            0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17,
            0.18, 0.19, 0.20,
            # right_thumb_y
            0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17,
            0.18, 0.19, 0.20,
            # left_elbow_x
            0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0064, 0.0128, 0.0256, 0.0512, 0.1024, 0.2048, 0.1024, 0.0512,
            0.0256, 0.0128, 0.0064, 0.0032, 0.0016, 0.0008, 0.0004,
            # left_elbow_y
            0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0064, 0.0128, 0.0256, 0.0512, 0.1024, 0.2048, 0.1024, 0.0512,
            0.0256, 0.0128, 0.0064, 0.0032, 0.0016, 0.0008, 0.0004,
            # right_elbow_x
            0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17,
            0.18, 0.19, 0.20,
            # right_elbow_y
            0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.30, 0.33, 0.36, 0.39, 0.42, 0.45, 0.48, 0.51,
            0.54, 0.57, 0.60,
            # left_shoulder_x
            0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17,
            0.18, 0.19, 0.20,
            # left_shoulder_y
            0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.30, 0.33, 0.36, 0.39, 0.42, 0.45, 0.48, 0.51,
            0.54, 0.57, 0.60,
            # right_shoulder_x
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0,
            # right_shoulder_y
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0,

            # distance_thumb_right_mouth_right_x
            0.49, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41, 0.40, 0.39, 0.38, 0.37, 0.36, 0.35, 0.34, 0.33,
            0.32, 0.31, 0.30,
            # distance_thumb_right_mouth_right_y
            0.49, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41, 0.40, 0.39, 0.38, 0.37, 0.36, 0.35, 0.34, 0.33,
            0.32, 0.31, 0.30,
            # distance_thumb_left_mouth_left_x
            0.5 - 0.3, 0.5 - 0.3002, 0.5 - 0.3004, 0.5 - 0.3008, 0.5 - 0.3016, 0.5 - 0.3032, 0.5 - 0.3064,
            0.5 - 0.3128, 0.5 - 0.3256, 0.5 - 0.3512, 0.5 - 0.4024, 0.5 - 0.5048, 0.5 - 0.4024, 0.5 - 0.3512,
            0.5 - 0.3256, 0.5 - 0.3128, 0.5 - 0.3064, 0.5 - 0.3032, 0.5 - 0.3016, 0.5 - 0.3008,

            # distance_thumb_left_mouth_left_y
            0.5 - 0.3, 0.5 - 0.3002, 0.5 - 0.3004, 0.5 - 0.3008, 0.5 - 0.3016, 0.5 - 0.3032, 0.5 - 0.3064,
            0.5 - 0.3128, 0.5 - 0.3256, 0.5 - 0.3512, 0.5 - 0.4024, 0.5 - 0.5048, 0.5 - 0.4024, 0.5 - 0.3512,
            0.5 - 0.3256, 0.5 - 0.3128, 0.5 - 0.3064, 0.5 - 0.3032, 0.5 - 0.3016, 0.5 - 0.3008,
        ]
        self.scaler_mock.get_scale_value = MagicMock(
            return_value=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        frames = pd.read_csv("test_processing.csv")
        expected_result = np.array([
            first_row
        ])
        data_processor = DataPreprocessor(data_scaler=self.scaler_mock, relative_to_first_frame=True)

        result = data_processor.preprocess_data(frames[:21])

        np.testing.assert_array_almost_equal(result, expected_result, decimal=10)

    def test_given_relative_to_first_frame_with_scaling(self):
        first_row = [
            # left_pinky_x
            0.01, 0.002, 0.03, 0.004, 0.05, 0.006, 0.07, 0.008, 0.09, 0.010, 0.11, 0.012, 0.13, 0.014, 0.15, 0.016,
            0.17, 0.018, 0.19, 0.020,
            # left_pinky_y
            0.01, 0.002, 0.03, 0.004, 0.05, 0.006, 0.07, 0.008, 0.09, 0.010, 0.11, 0.012, 0.13, 0.014, 0.15, 0.016,
            0.17, 0.018, 0.19, 0.020,
            # right_pinky_x
            0.0002, 0.00004, 0.0008, 0.00016, 0.0032, 0.00064, 0.0128, 0.00256, 0.0512, 0.01024, 0.2048, 0.01024,
            0.0512, 0.00256, 0.0128, 0.00064, 0.0032, 0.00016, 0.0008, 0.00004,
            # right_pinky_y
            0.0002, 0.00004, 0.0008, 0.00016, 0.0032, 0.00064, 0.0128, 0.00256, 0.0512, 0.01024, 0.2048, 0.01024,
            0.0512, 0.00256, 0.0128, 0.00064, 0.0032, 0.00016, 0.0008, 0.00004,
            # left_index_x
            0.01, 0.002, 0.03, 0.004, 0.05, 0.006, 0.07, 0.008, 0.09, 0.010, 0.11, 0.012, 0.13, 0.014, 0.15, 0.016,
            0.17, 0.018, 0.19, 0.020,
            # left_index_y
            0.03, 0.006, 0.09, 0.012, 0.15, 0.018, 0.21, 0.024, 0.27, 0.030, 0.33, 0.036, 0.39, 0.042, 0.45, 0.048,
            0.51, 0.054, 0.57, 0.060,
            # right_index_x
            0.0002, 0.00004, 0.0008, 0.00016, 0.0032, 0.00064, 0.0128, 0.00256, 0.0512, 0.01024, 0.2048, 0.01024,
            0.0512, 0.00256, 0.0128, 0.00064, 0.0032, 0.00016, 0.0008, 0.00004,
            # right_index_y
            0.0002, 0.00004, 0.0008, 0.00016, 0.0032, 0.00064, 0.0128, 0.00256, 0.0512, 0.01024, 0.2048, 0.01024,
            0.0512, 0.00256, 0.0128, 0.00064, 0.0032, 0.00016, 0.0008, 0.00004,
            # left_thumb_x
            0.0002, 0.00004, 0.0008, 0.00016, 0.0032, 0.00064, 0.0128, 0.00256, 0.0512, 0.01024, 0.2048, 0.01024,
            0.0512, 0.00256, 0.0128, 0.00064, 0.0032, 0.00016, 0.0008, 0.00004,
            # left_thumb_y
            0.0002, 0.00004, 0.0008, 0.00016, 0.0032, 0.00064, 0.0128, 0.00256, 0.0512, 0.01024, 0.2048, 0.01024,
            0.0512, 0.00256, 0.0128, 0.00064, 0.0032, 0.00016, 0.0008, 0.00004,
            # right_thumb_x
            0.01, 0.002, 0.03, 0.004, 0.05, 0.006, 0.07, 0.008, 0.09, 0.010, 0.11, 0.012, 0.13, 0.014, 0.15, 0.016,
            0.17, 0.018, 0.19, 0.020,
            # right_thumb_y
            0.01, 0.002, 0.03, 0.004, 0.05, 0.006, 0.07, 0.008, 0.09, 0.010, 0.11, 0.012, 0.13, 0.014, 0.15, 0.016,
            0.17, 0.018, 0.19, 0.020,
            # left_elbow_x
            0.0002, 0.00004, 0.0008, 0.00016, 0.0032, 0.00064, 0.0128, 0.00256, 0.0512, 0.01024, 0.2048, 0.01024,
            0.0512, 0.00256, 0.0128, 0.00064, 0.0032, 0.00016, 0.0008, 0.00004,
            # left_elbow_y
            0.0002, 0.00004, 0.0008, 0.00016, 0.0032, 0.00064, 0.0128, 0.00256, 0.0512, 0.01024, 0.2048, 0.01024,
            0.0512, 0.00256, 0.0128, 0.00064, 0.0032, 0.00016, 0.0008, 0.00004,
            # right_elbow_x
            0.01, 0.002, 0.03, 0.004, 0.05, 0.006, 0.07, 0.008, 0.09, 0.010, 0.11, 0.012, 0.13, 0.014, 0.15, 0.016,
            0.17, 0.018, 0.19, 0.020,
            # right_elbow_y
            0.03, 0.006, 0.09, 0.012, 0.15, 0.018, 0.21, 0.024, 0.27, 0.030, 0.33, 0.036, 0.39, 0.042, 0.45, 0.048,
            0.51, 0.054, 0.57, 0.060,
            # left_shoulder_x
            0.01, 0.002, 0.03, 0.004, 0.05, 0.006, 0.07, 0.008, 0.09, 0.010, 0.11, 0.012, 0.13, 0.014, 0.15, 0.016,
            0.17, 0.018, 0.19, 0.020,
            # left_shoulder_y
            0.03, 0.006, 0.09, 0.012, 0.15, 0.018, 0.21, 0.024, 0.27, 0.030, 0.33, 0.036, 0.39, 0.042, 0.45, 0.048,
            0.51, 0.054, 0.57, 0.060,
            # right_shoulder_x
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0,
            # right_shoulder_y
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0,

            # # distance_thumb_right_mouth_right_x
            # 0.49, 0.048, 0.47, 0.046, 0.45, 0.044, 0.43, 0.042, 0.41, 0.040, 0.39, 0.038, 0.37, 0.036, 0.35, 0.034,
            # 0.33, 0.032, 0.31, 0.030,
            # # distance_thumb_right_mouth_right_y
            # 0.49, 0.048, 0.47, 0.046, 0.45, 0.044, 0.43, 0.042, 0.41, 0.040, 0.39, 0.038, 0.37, 0.036, 0.35, 0.034,
            # 0.33, 0.032, 0.31, 0.030,
            # # distance_thumb_left_mouth_left_x
            # 0.5 - 0.3, (0.5 - 0.3002) * 0.1, 0.5 - 0.3004, (0.5 - 0.3008) * 0.1, 0.5 - 0.3016, (0.5 - 0.3032) * 0.1,
            # 0.5 - 0.3064, (0.5 - 0.3128) * 0.1, 0.5 - 0.3256, (0.5 - 0.3512) * 0.1, 0.5 - 0.4024, (0.5 - 0.5048) * 0.1,
            # 0.5 - 0.4024, (0.5 - 0.3512) * 0.1, 0.5 - 0.3256, (0.5 - 0.3128) * 0.1, 0.5 - 0.3064, (0.5 - 0.3032) * 0.1,
            # 0.5 - 0.3016, (0.5 - 0.3008) * 0.1,
            #
            # # distance_thumb_left_mouth_left_y
            # 0.5 - 0.3, (0.5 - 0.3002) * 0.1, 0.5 - 0.3004, (0.5 - 0.3008) * 0.1, 0.5 - 0.3016, (0.5 - 0.3032) * 0.1,
            # 0.5 - 0.3064, (0.5 - 0.3128) * 0.1, 0.5 - 0.3256, (0.5 - 0.3512) * 0.1, 0.5 - 0.4024, (0.5 - 0.5048) * 0.1,
            # 0.5 - 0.4024, (0.5 - 0.3512) * 0.1, 0.5 - 0.3256, (0.5 - 0.3128) * 0.1, 0.5 - 0.3064, (0.5 - 0.3032) * 0.1,
            # 0.5 - 0.3016, (0.5 - 0.3008) * 0.1,

            # distance_thumb_right_mouth_right_x
            0.49, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41, 0.40, 0.39, 0.38, 0.37, 0.36, 0.35, 0.34, 0.33,
            0.32, 0.31, 0.30,
            # distance_thumb_right_mouth_right_y
            0.49, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41, 0.40, 0.39, 0.38, 0.37, 0.36, 0.35, 0.34, 0.33,
            0.32, 0.31, 0.30,
            # distance_thumb_left_mouth_left_x
            0.5 - 0.3, 0.5 - 0.3002, 0.5 - 0.3004, 0.5 - 0.3008, 0.5 - 0.3016, 0.5 - 0.3032, 0.5 - 0.3064,
            0.5 - 0.3128, 0.5 - 0.3256, 0.5 - 0.3512, 0.5 - 0.4024, 0.5 - 0.5048, 0.5 - 0.4024, 0.5 - 0.3512,
            0.5 - 0.3256, 0.5 - 0.3128, 0.5 - 0.3064, 0.5 - 0.3032, 0.5 - 0.3016, 0.5 - 0.3008,

            # distance_thumb_left_mouth_left_y
            0.5 - 0.3, 0.5 - 0.3002, 0.5 - 0.3004, 0.5 - 0.3008, 0.5 - 0.3016, 0.5 - 0.3032, 0.5 - 0.3064,
            0.5 - 0.3128, 0.5 - 0.3256, 0.5 - 0.3512, 0.5 - 0.4024, 0.5 - 0.5048, 0.5 - 0.4024, 0.5 - 0.3512,
            0.5 - 0.3256, 0.5 - 0.3128, 0.5 - 0.3064, 0.5 - 0.3032, 0.5 - 0.3016, 0.5 - 0.3008,
        ]
        self.scaler_mock.get_scale_value = MagicMock(
            return_value=[1, 0.1, 1, 0.1, 1, 0.1, 1, 0.1, 1, 0.1, 1, 0.1, 1, 0.1, 1, 0.1, 1, 0.1, 1, 0.1])
        frames = pd.read_csv("test_processing.csv")
        expected_result = np.array([
            first_row
        ])
        data_processor = DataPreprocessor(data_scaler=self.scaler_mock, relative_to_first_frame=True)

        result = data_processor.preprocess_data(frames[:21])

        np.testing.assert_array_almost_equal(result[:, :40], expected_result[:, 0:40])
        np.testing.assert_array_almost_equal(result[:, 40:80], expected_result[:, 40:80])
        np.testing.assert_array_almost_equal(result[:, 80:120], expected_result[:, 80:120])
        np.testing.assert_array_almost_equal(result[:, 120:160], expected_result[:, 120:160])
        np.testing.assert_array_almost_equal(result[:, 160:200], expected_result[:, 160:200])
        np.testing.assert_array_almost_equal(result[:, 200:240], expected_result[:, 200:240])
        np.testing.assert_array_almost_equal(result[:, 240:280], expected_result[:, 240:280])
        np.testing.assert_array_almost_equal(result[:, 280:320], expected_result[:, 280:320])
        np.testing.assert_array_almost_equal(result[:, 280:320], expected_result[:, 280:320])
        np.testing.assert_array_almost_equal(result[:, 320:400], expected_result[:, 320:400])
        np.testing.assert_array_almost_equal(result[:, 400:420], expected_result[:, 400:420])
        np.testing.assert_array_almost_equal(result[:, 420:440], expected_result[:, 420:440])
        np.testing.assert_array_almost_equal(result, expected_result, decimal=10)


if __name__ == '__main__':
    unittest.main()
