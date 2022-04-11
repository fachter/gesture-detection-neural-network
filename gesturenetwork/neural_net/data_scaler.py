import numpy as np


class DataScaler:
    def __init__(self):
        pass

    def scale_data(self, left_shoulder_x, left_shoulder_y, right_shoulder_x, right_shoulder_y, data_to_scale):
        # x_diff = right_shoulder_x - left_shoulder_x
        # y_diff = right_shoulder_y - left_shoulder_y
        # scale_factor = np.sqrt(x_diff ** 2 + y_diff ** 2) + 1e-6
        # kehrbruch = 1. / scale_factor
        result = np.round(np.multiply(data_to_scale,
                                      self.get_scale_value(left_shoulder_x, left_shoulder_y,
                                                           right_shoulder_x, right_shoulder_y)[:, np.newaxis]), 3)
        return result

    @staticmethod
    def get_scale_value(left_shoulder_x, left_shoulder_y, right_shoulder_x, right_shoulder_y):
        x_diff = right_shoulder_x - left_shoulder_x
        y_diff = right_shoulder_y - left_shoulder_y
        scale_factor = np.sqrt(x_diff ** 2 + y_diff ** 2) + 1e-6
        kehrbruch = 1. / scale_factor
        return kehrbruch
