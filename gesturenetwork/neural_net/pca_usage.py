import pandas as pd
import numpy as np
from netneural.pca.pca import PCA
from .data_preprocessor import DataPreprocessor
import matplotlib.pyplot as plt


def create_preprocessed_attributes(keys, tags, frames):
    out = list()
    for key in keys:
        for tag in tags:
            for i in range(frames):
                out.append(key + tag + str(i))
    return out


if __name__ == '__main__':
    # TODO: not running anymore (is it still necessary to exist?)
    # VISUALIZE DATA
    frames = pd.read_csv("../data/data_merge/merged_gesture_data.csv")
    data = frames.drop(columns=['ground_truth', 'Unnamed: 0', 'timestamp'])
    left_pinky_x = [x for x in frames['left_pinky_x'][:500]]
    gt = [x for x in frames['ground_truth']]
    unique_labels, inverse = np.unique(gt, return_inverse=True)

    # we can only look at one dimension at a time
    # plot_scatter(left_pinky_x, list(range(500)), inverse[:500], 'time', 'left_pinky_x', 'Left Pinky X Coordinate')

    # perform PCA
    pca = PCA()
    eigenvectors, eigenvalues = pca.get_eigenvectors_and_values(data)  # reduce data to 100 principal components
    reduced_data = pca.get_n_dimensions(data, 100, eigenvectors)
    ordered_attributes_matrix = pca.analyze_eigenvectors(eigenvectors, list(data.columns))
    ordered_attributes = pca.compute_order_attributes(ordered_attributes_matrix)
    n_pc = pca.explained_variance(eigenvalues, 0.991, False)

    # look at first four principal components data
    pc1 = reduced_data[:, 0]
    pc2 = reduced_data[:, 1]
    pc3 = reduced_data[:, 2]
    pc4 = reduced_data[:, 3]

    # data can be visualized using 4 dimensions
    plt.figure(figsize=(20, 10))
    ax = plt.axes(projection='3d')
    ax.scatter3D(pc1, pc2, pc3, c=inverse, s=pc4)
    plt.show()

    # or using 3 dimensions
    plt.scatter(pc1, pc2, s=pc3, c=inverse, alpha=0.5)
    plt.show()

    # analyze how much variance is captured using n principal components
    pca.explained_variance(eigenvalues)

    # SAME WITH PREPROCESSED DATA
    data_preprocessor = DataPreprocessor()
    X = data_preprocessor.preprocess_data(data[:1020], including_ground_truth=False)
    eigenvectors, eigenvalues = pca.get_eigenvectors_and_values(X)
    reduced_data = pca.get_n_dimensions(X, 100, eigenvectors)
    keys = ["left_pinky", "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb"]
    keys = ["left_pinky", "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb"]
    attributes_preprocessed = create_preprocessed_attributes(keys, ['_x', '_y'], 20)
    # ordered_attributes = analyze_eigenvector(eigenvectors[:,0], attributes_preprocessed)

    pc1 = reduced_data[:, 0]
    pc2 = reduced_data[:, 1]
    pc3 = reduced_data[:, 2]
    pc4 = reduced_data[:, 3]
    inverse = inverse[:1000]

    plt.figure(figsize=(20, 10))
    ax = plt.axes(projection='3d')
    ax.scatter3D(pc1, pc2, pc3, c=inverse, s=pc4)
    plt.show()

    plt.scatter(pc1, pc2, s=pc3, c=inverse, alpha=0.5)
    plt.show()

    pca.explained_variance(eigenvalues)
