import lzma
import pickle
from datetime import datetime
import json
from typing import List

import numpy as np

from netneural.network.feature_scaling import StandardScaler
from netneural.network.nn import NeuralNetwork
from netneural.network.one_hot_encoder import OneHotEncoder
from netneural.pca.pca import PCA


def load_session_from_config(file_name, histories=False, train=False):
    """
    Loads the parameters from a training session of a neural net.
    :param histories: if histories should be outputted as well
    :param file_name: the .json file that stores the parameters of the training session
    :return: the neural network, the pca object, the f1 score history and the used learning rate for training that
    specific neural network
    """
    if file_name.endswith('.json'):
        with open(file_name) as json_file:
            data = json.load(json_file)
    else:
        with lzma.open(file_name, "rb") as fin:
            data = pickle.load(fin)
    try:
        regression = data['regression']
    except KeyError:
        regression = False
    try:
        f1_score_history = data['f1_score_history']
    except KeyError:
        f1_score_history = None
    try:
        included_frames = data['included_frames']
    except KeyError:
        included_frames = 20
    # try:
    #     pca_n = data['pca_n']
    # except KeyError:
    #     pca_n = None
    # try:
    #     unique_labels = data['unique_labels']
    #     encoder = OneHotEncoder(unique_labels=unique_labels)
    # except KeyError:
    #     encoder = None

    # weights = []
    # for weight in data['weights']:
    #     weights.append(np.array(weight))
    # neural_network = NeuralNetwork(tuple(data['shape']),
    #                                activation_function=data['activation_function'],
    #                                weight_matrices=weights,
    #                                encoder=encoder,
    #                                regression=regression)

    # if pca_n is not None:
    #     pca = PCA(data['pca_n'], np.array(data['eigenvectors']))
    # else:
    #     pca = None

    # get scaler from training data
    #scaler = StandardScaler()  # for now Standard Scaler is always used
    try:
        X_train = np.array(data['X_train'])
    except KeyError:
        X_train = None
    #scaler.fit(X_train)

    try:
        lr = data['learning_rate']
    except:
        lr = None
    try:
        y_train = data['y_train']
    except KeyError:
        y_train = None
    try:
        X_test = data['X_test']
    except KeyError:
        X_test = None
    try:
        y_test = data['y_test']
    except KeyError:
        y_test = None
    if train:
        return included_frames, f1_score_history, lr, X_train, np.array(y_train), \
               np.array(X_test), np.array(y_test), data['acc_history'], data['error_history']

    if histories:
        if regression:
            return included_frames, data['error_history']
        else:
            return included_frames, data['f1_score_history'], data['acc_history'], data['error_history']
    return included_frames, f1_score_history, lr, X_train, np.array(y_train), \
           np.array(X_test), np.array(y_test)


def save_session(nn: NeuralNetwork, learning_rate: float, X_train, y_train, X_test, y_test, iterations, error_history,
                 unique_labels: np.ndarray = None, pca: PCA = None, f1_score_history: List[float] = None,
                 accuracy_history=None, included_frames=20):
    """
    Saves parameters of a training session of a neural network to a file.
    :param nn: the trained neural network instance
    :param unique_labels: the different possible output classes
    :param pca: the pca instance storing eigenvectors, and number of principal components
    :param f1_score_history: history of the f1 score over the time of training on the test data set
    :param learning_rate: used alpha for training
    :return:
    """
    json_weights = convert_np_to_json(nn.weights)
    json_init_weights = convert_np_to_json(nn.init_weights)
    json_eigenvectors = convert_np_to_json(pca.eigenvectors) if pca is not None else None
    json_X_train = convert_np_to_json(X_train)
    json_y_train = convert_np_to_json(y_train)
    json_X_test = convert_np_to_json(X_test)
    json_y_test = convert_np_to_json(y_test)
    if not isinstance(unique_labels, list) and unique_labels is not None:
        unique_labels = unique_labels.tolist()
    timestamp_string = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    if nn.regression:
        fname = f'saved_configs/config_{round(error_history[-1] * 100)}_f1_{timestamp_string}.pkl'
    else:
        fname = f'saved_configs/config_{round(f1_score_history[-1] * 100)}_f1_{timestamp_string}.pkl'
    config = {
        'shape': nn.shape,
        'regression': nn.regression,
        'activation_function': nn.activation_function.__name__,
        'unique_labels': unique_labels,
        'included_frames': included_frames,
        'pca_n': pca.n if pca is not None else None,
        'learning_rate': learning_rate,
        'iterations': iterations,
        'optimizer': str(nn.optimizer),
        'weights': json_weights,
        'initial_weights': json_init_weights,
        'f1_score_history': f1_score_history,
        'eigenvectors': json_eigenvectors,
        'X_train': json_X_train,
        'y_train': json_y_train,
        'X_test': json_X_test,
        'y_test': json_y_test,
        'acc_history': accuracy_history,
        'error_history': error_history
    }

    with lzma.open(fname, "wb") as fout:
        pickle.dump(config, fout)

    # with open(fname, 'w') as file:
    #    json.dump(config, file)
    print(f'Session Configuration stored in file {fname}')


def convert_np_to_json(np_array):
    json_weights = []
    for weight in np_array:
        json_weights.append(weight.tolist())
    return json_weights
