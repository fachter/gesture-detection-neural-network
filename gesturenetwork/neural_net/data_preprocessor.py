import numpy as np

from .data_scaler import DataScaler


class DataPreprocessor:
    def __init__(self, including_frames=20, data_scaler=None, relative_to_first_frame=False, percentage_majority=.5):
        self.including_frames = including_frames
        self.relative_to_first_frame = relative_to_first_frame
        self.percentage_majority = percentage_majority
        if data_scaler is None:
            self.data_scaler = DataScaler()
        else:
            self.data_scaler = data_scaler

    def preprocess_data(self, frames, including_ground_truth=False):
        x = []
        keys = ["left_pinky", "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb",
                "left_elbow", "right_elbow", "left_shoulder", "right_shoulder"]
        y = []

        for index in range(len(frames) - self.including_frames):
            row = []
            scale_vector = self.data_scaler.get_scale_value(
                frames["left_shoulder_x"][index:index + self.including_frames],
                frames["left_shoulder_y"][index:index + self.including_frames],
                frames["right_shoulder_x"][index:index + self.including_frames],
                frames["right_shoulder_y"][index:index + self.including_frames],
            )
            for key in keys:
                x_values, y_values = self.get_x_and_y_values_for_key(frames, index, key)
                scaled_x = np.multiply(x_values, scale_vector)
                scaled_y = np.multiply(y_values, scale_vector)
                row.extend(scaled_x)
                row.extend(scaled_y)
            row.extend(self.get_distances_hand_to_mouth(frames, index))
            x.append(row)
            if including_ground_truth:
                y.append(self.get_most_occuring_ground_truth(frames, index))
        if including_ground_truth:
            return np.array(x), np.array(y)
        return np.array(x)

    def get_most_occuring_ground_truth(self, frames, index):
        unique, pos = np.unique(frames["ground_truth"][index:index + self.including_frames], return_inverse=True)
        counts = np.bincount(pos)
        if (counts[counts.argmax()] / self.including_frames) > self.percentage_majority:
            maxpos = counts.argmax()
            ground_truth = unique[maxpos]
            return ground_truth
        return "idle"

    def get_x_and_y_values_for_key(self, frames, index, key):
        x_key = key + "_x"
        y_key = key + "_y"
        x_value = frames[x_key]
        y_value = frames[y_key]
        previous_x = x_value.iloc[index]
        previous_y = y_value.iloc[index]
        x_values = []
        y_values = []
        for i in range(index + 1, index + self.including_frames + 1):
            x_values.append(x_value.iloc[i] - previous_x)
            y_values.append(y_value.iloc[i] - previous_y)
            if not self.relative_to_first_frame:
                previous_x = x_value.iloc[i]
                previous_y = y_value.iloc[i]
        return x_values, y_values

    def get_distances_hand_to_mouth(self, frames, index):
        distances = []
        distances_right_x = []
        distances_right_y = []
        distances_left_x = []
        distances_left_y = []
        for i in range(index, index + self.including_frames):
            distances_right_x.append(frames["right_mouth_x"].iloc[i] - frames["right_thumb_x"].iloc[i])
            distances_right_y.append(frames["right_mouth_y"].iloc[i] - frames["right_thumb_y"].iloc[i])
            distances_left_x.append(frames["left_mouth_x"].iloc[i] - frames["left_thumb_x"].iloc[i])
            distances_left_y.append(frames["left_mouth_y"].iloc[i] - frames["left_thumb_y"].iloc[i])
        distances.extend(distances_right_x)
        distances.extend(distances_right_y)
        distances.extend(distances_left_x)
        distances.extend(distances_left_y)
        return distances
