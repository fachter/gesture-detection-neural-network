import pandas as pd

from .data_preprocessor import DataPreprocessor
from .data_scaler import DataScaler
from netneural.network.one_hot_encoder import OneHotEncoder
from netneural.pca.pca import PCA


class DataProcessor:
    def __init__(self, preprocessor_frames=20, pca: PCA = None, encoder: OneHotEncoder = None):
        self.preprocessor = DataPreprocessor(preprocessor_frames)
        self.pca = pca
        self.encoder = encoder
        self.scaler = DataScaler()

    def full_processing_combined(self, data: pd.DataFrame, including_ground_truth=False):
        x, y = self.preprocessor.preprocess_data(data, including_ground_truth)

        scaled_x = self.scaler.scale_data(data["left_shoulder_x"].to_numpy(), data["left_shoulder_y"].to_numpy(),
                                          data["right_shoulder_x"].to_numpy(), data["right_shoulder_y"].to_numpy(),
                                          data_to_scale=x)

        if self.encoder is None:
            self.encoder = OneHotEncoder()
        y_one_hot = self.encoder.encode(y)

        return scaled_x, y_one_hot

        # if self.pca is None:
        #     X, y_one_hot, X_test, y_test = train_test_split(scaled_x, y_one_hot, 0.8, 0.2, randomized=True)[:4]
        #     self.pca = PCA()
        #     X = self.pca.pca(X)
        #     X_test = self.pca.transform_data(X_test)
        #     return X, y_one_hot, X_test, y_test
        # else:
        #     # this means, that the data comes from live mode
        #     return self.pca.transform_data(scaled_x)
