import os
import random

from netneural.session.data_loader import train_test_split, train_test_split_idxs
from netneural.network.nn import NeuralNetwork
import numpy as np
from netneural.network.one_hot_encoder import OneHotEncoder
from netneural.pca.pca import PCA
from netneural.session.nn_session_util import save_session as save_network_config
from netneural.session.nn_session_util import load_from_config
from .data_preprocessor import DataPreprocessor
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

from .session_util import save_session, load_session_from_config

np.set_printoptions(threshold=sys.maxsize)


def main(including_frames=20):
    folder_name = "../data/data_merge/output_csv/optional/rotate_left/"
    files = os.listdir(folder_name)
    frames = None
    for i in range(len(files)):
        if not os.path.isfile(folder_name + files[i]):
            continue
        if frames is None:
            frames = pd.read_csv(folder_name + files[i])
        else:
            frames = frames.append(pd.read_csv(folder_name + files[i]), ignore_index=True)
    data_preprocessor = DataPreprocessor(including_frames, relative_to_first_frame=True, percentage_majority=.7)
    X, y = data_preprocessor.preprocess_data(frames, including_ground_truth=True)
    with open(f'preprocessed_data/combined_data_{including_frames}_frames_optional_rl.npy', 'wb') as f:
        np.save(f, X)
        np.save(f, y)
    unique, counts = np.unique(y, return_counts=True)
    print(unique)
    print(counts)
    # train_model_with_params(X, y, iterations, learning_rate, activation_function, including_frames)
    # train_model_with_params(X, y, 10000, 0.3)
    # train_model_with_params(X, y, 10000, 0.3, "relu")
    # train_model_with_params(X, y, 10000, 0.03)
    # train_model_with_params(X, y, 20000, 0.03)
    # train_model_with_params(X, y, 20000, 0.3)


def distribute_equally(X, y: np.array, factor=1., optional=False):
    unique, counts = np.unique(y, return_counts=True)
    idxs = dict()
    for label in unique:
        idxs[label] = np.where(y == label)

    idle_idxs = idxs['idle'][0]
    random.shuffle(idle_idxs)
    if optional:
        counts[0], counts[1] = counts[1], counts[0]
        unique[0], unique[1] = unique[1], unique[0]
    idle_pos = 0
    max_idle = max(np.delete(counts, idle_pos))
    max_idle = min(len(idle_idxs), max_idle * factor)
    idle_idxs = idle_idxs[:max_idle]

    X_equal = X[idle_idxs]
    y_equal = y[idle_idxs]
    for label in unique[1:]:
        X_equal = np.append(X_equal, X[idxs[label][0]], axis=0)
        y_equal = np.append(y_equal, y[idxs[label][0]], axis=0)

    return X_equal, y_equal


def train_model_with_params(X, y, iterations, learning_rate, shape, included_frames, activation_function="sigmoid",
                            batch_size=None,
                            uniform=False, optional=False, factor=1.):
    print('#### TRAINING THE NETWORK ####')
    if uniform:
        X, y = distribute_equally(X, y, optional=optional, factor=factor)

    encoder = OneHotEncoder()
    y_one_hot = encoder.encode(y).T
    print('Following unique labels were found:' + str(encoder.unique_labels) + '\n')

    # split in training and test data
    train_idxs, test_idxs = train_test_split_idxs(len(X), 0.8)
    X_train = X[train_idxs]
    y_train = y_one_hot[:, train_idxs]
    X_test = X[test_idxs]
    y_test = y_one_hot[:, test_idxs]
    # X_train, y_train, X_test, y_test = train_test_split(X, y_one_hot, 0.8, 0.2, randomized=True)[:4]

    print('Counts for each class:')
    unique, counts = np.unique(encoder.decode(y_one_hot.T), return_counts=True)
    print(unique)
    print(counts)
    unique_test, counts_test = np.unique(encoder.decode(y_test.T), return_counts=True)
    print(unique_test)
    print(counts_test)
    print('\n')

    # perform pca
    print(f'PCA reduced attribute count from {X.shape[1]}')
    pca = PCA()
    X_train = pca.pca(X_train, var_per=0.999)
    X_test = pca.transform_data(X_test)
    print(f'to {X_train.shape[1]} \n')

    print('#### STARTING TRAINING ####')
    hidden_layer_1 = shape[0]
    hidden_layer_2 = shape[1]
    hidden_layer_3 = shape[2]
    shape = (X_train.shape[1], 64, 32, 16, y_one_hot.shape[0])
    print("Shape:", shape)
    print("Learning rate:", learning_rate)
    print('Iterations:', iterations)
    neural_network = NeuralNetwork(shape, activation_function=activation_function, encoder=encoder)

    # scale data
    neural_network.scaler.fit(X_train)  # standard scaler by default
    X_scaled = neural_network.scaler.transform(X_train)
    X_test_scaled = neural_network.scaler.transform(X_test)

    f1_history, acc_history, err_history = neural_network.train(X_scaled, y_train, iterations, learning_rate,
                                                                test_data=X_test_scaled, test_target=y_test,
                                                                optimizer='adam', plots=True, batch_size=batch_size)
    predictions_distribution = neural_network.forward_pass(X_test_scaled)
    truth = np.argmax(y_test, axis=0)
    confusion_matrix = np.zeros((y_test.shape[0], y_test.shape[0]))
    predictions_index = np.argmax(predictions_distribution, axis=0)
    for i in range(len(predictions_index)):
        confusion_matrix[truth[i] - 1][predictions_index[i] - 1] += 1
    # sns.heatmap(confusion_matrix)

    y_actu = pd.Series(neural_network.encoder.decode(y_test.T), name='Actual')
    y_pred = pd.Series(neural_network.predict(X_test_scaled), name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'])
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.set(font_scale=2)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30, rotation=90)
    # if batch_size is None:
    #     ax.set_title('No Batch Training')
    # else:
    #     ax.set_title(f'Batchsize: {batch_size}')
    plt.xlabel('Prediction', fontsize=40)
    plt.ylabel('Actual', fontsize=40)
    heatmap = sns.heatmap(df_confusion, annot=True, ax=ax, fmt='d')

    # cax = plt.gcf().axes[-1]
    # cax.tick_params(labelsize=60)

    plt.show()
    per_class_accuracy = np.diag(df_confusion) / (df_confusion.sum(axis=1) + 1e-9)
    print(f'Accuracy per class: {(per_class_accuracy * 100).round(2)}\n')
    total_acc = (per_class_accuracy.mean() * 100).round(2)
    print(f'Total Accuracy: {total_acc}')
    print(f'Total F1 Score: {f1_history[-1]}')
    save_network_config(neural_network, neural_network.encoder.unique_labels, pca, f1_history[-1],
                        'saved_configs/mandatory')
    save_session(neural_network, learning_rate, X_train, y_train, X_test, y_test, iterations, err_history,
                 f1_score_history=f1_history, accuracy_history=acc_history, included_frames=included_frames,
                 pca=pca, unique_labels=neural_network.encoder.unique_labels)
    back_to_labels = encoder.decode(predictions_distribution.T)


def train_model_from_session(config_file, iterations, batch_size=None):
    frames, f1_score_history, learning_rate, X_train, y_train, X_test, y_test, acc_history, err_history \
        = load_session_from_config(config_file, train=True)
    nn, pca = load_from_config(config_file)

    # Scale data
    nn.scaler.fit(X_train)
    X_scaled = nn.scaler.transform(X_train)
    X_test_scaled = nn.scaler.transform(X_test)

    print('#### TRAINING THE NETWORK ####')
    f1_score_history_new, acc_history_new, err_history_new = nn.train(X_scaled, y_train, iterations, learning_rate,
                                                                      test_data=X_test_scaled,
                                                                      test_target=y_test,
                                                                      optimizer="adam", batch_size=batch_size,
                                                                      plots=True)

    f1_score_history = f1_score_history + f1_score_history_new
    acc_history = acc_history + acc_history_new
    err_history = err_history + err_history_new

    predictions_distribution = nn.forward_pass(X_test_scaled)
    truth = np.argmax(y_test, axis=0)
    confusion_matrix = np.zeros((y_test.shape[0], y_test.shape[0]))
    predictions_index = np.argmax(predictions_distribution, axis=0)
    for i in range(len(predictions_index)):
        confusion_matrix[truth[i]][predictions_index[i]] += 1

    y_actu = pd.Series(nn.encoder.decode(y_test.T), name='Actual')
    y_pred = pd.Series(nn.predict(X_test_scaled), name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.set(font_scale=2)
    heatmap = sns.heatmap(df_confusion, annot=True, ax=ax, fmt='d')
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=20)
    # heatmap.set_xlabel(heatmap.get_xlabel(), font_size=20)
    # heatmap.set_ylabel(font_size=20)
    # sns.heatmap(cf_matrix, annot=labels, fmt=‘’, cmap = 'Blues')
    #
    # labels = np.array([
    #     []
    # ])
    # sns.heatmap(confusion_matrix, annot=labels)
    plt.show()
    per_class_accuracy = np.diag(confusion_matrix) / (confusion_matrix.sum(axis=1) + 1e-9)
    print(f'Accuracy per class: {(per_class_accuracy * 100).round(2)}\n')
    total_acc = (per_class_accuracy.mean() * 100).round(2)
    print(f'Total Accuracy: {total_acc}')
    print(f'Total F1 Score: {f1_score_history[-1]}')
    # save_network_config(nn, nn.encoder.unique_labels, pca, f1_score_history[-1],
    #                    'saved_configs/mandatory/rel_70')
    # save_session(nn, learning_rate, X_train, y_train, X_test, y_test, iterations, err_history,
    #              f1_score_history=f1_score_history, accuracy_history=acc_history, included_frames=frames,
    #              pca=pca, unique_labels=nn.encoder.unique_labels)
    back_to_labels = nn.encoder.decode(predictions_distribution.T)


if __name__ == "__main__":
    # for frames in [20, 25]:
    #    print(f'Preprocessing {frames} frames')
    #    main(frames)

    noise = True
    optional = False
    new = True
    n = 25
    # rel = 'relative_to_first_and_60_percent_majority/'
    # rel = 'relative_to_first_and_70_percent_majority/'
    # rel_mand = 'relative_to_first_and_70_percent_majority/'
    rel = 'optional/combined_'
    # rel = 'mandatory/rel_70'
    rel_mand = 'mandatory/combined_'

    with open(f'preprocessed_data/{rel_mand}data_{n}_frames_mandatory.npy', 'rb') as file:
        # noinspection PyTypeChecker
        X = np.load(file)
        # noinspection PyTypeChecker
        y = np.load(file)
    if noise:
        with open(f'preprocessed_data/{rel_mand}data_{n}_frames_mandatory_noise.npy', 'rb') as file:
            X_merge = np.load(file)
            y_merge = np.load(file)
        X = np.append(X, X_merge, axis=0)
        y = np.append(y, y_merge, axis=0)
    if optional:
        with open(f'preprocessed_data/{rel}data_{n}_frames_optional.npy', 'rb') as file:
            X_merge = np.load(file)
            y_merge = np.load(file)
        X = np.append(X, X_merge, axis=0)
        y = np.append(y, y_merge, axis=0)
        # with open(f'preprocessed_data/{rel}data_{n}_frames_optional_rl.npy', 'rb') as file:
        #    X_merge = np.load(file)
        #    y_merge = np.load(file)
        # X = np.append(X, X_merge, axis=0)
        # y = np.append(y, y_merge, axis=0)
    if new:
        with open(f'preprocessed_data/{rel_mand}data_{n}_frames_mandatory_new.npy', 'rb') as file:
            X_merge = np.load(file)
            y_merge = np.load(file)
        X = np.append(X, X_merge, axis=0)
        y = np.append(y, y_merge, axis=0)

    train_model_with_params(X, y, 1500, 0.001, shape=(80, 80, 80), activation_function='sigmoid', included_frames=n,
                            batch_size=500, uniform=True, optional=optional, factor=4)

    # print(f'Used {n} frames')
    # train_model_from_session('saved_configs/optional/uniform/config_94_f1_2022-04-01 14-41-24.pkl', 1, batch_size=1000)
