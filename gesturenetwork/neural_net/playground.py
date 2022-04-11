import pandas as pd
import numpy as np
import json
from datetime import datetime
from netneural.network import nn

#
# def save_nn_config(neural_network: nn.NeuralNetwork, acc):
#     json_weights = []
#     for weight in neural_network.weights:
#         json_weights.append(weight.tolist())
#     config = {
#         'weights': json_weights,
#         'shape': neural_network.shape,
#         'activation_function': neural_network.activation_function.__name__
#     }
#
#     timestamp_string = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
#     with open(f'config_{round(acc * 100)}_accuracy_{timestamp_string}.json', 'w') as file:
#         json.dump(config, file)
#


def load_nn_from_config(file_name):
    with open(file_name) as json_file:
        data = json.load(json_file)
        weights = []
        for weight in data['weights']:
            weights.append(np.array(weight))
        neural_network = nn.NeuralNetwork(tuple(data['shape']),
                                          activation_function=data['activation_function'],
                                          weight_matrices=weights)
        return neural_network


def main():
    frames = pd.DataFrame(pd.read_csv(
        "/Users/felixachter/source/uni/machine-learning/neural-network-framework/"
        "data/data_merge/output_csv/rotate_felix_gt.csv"))

    X = frames[get_keys_to_use(frames)].to_numpy()
    y = frames["ground_truth"].to_numpy()
    y[y == "idle"] = 0
    y[y == "rotate"] = 1
    x_with_20_frames_in_past = []
    ground_truths = []
    for index in range(len(X) - 20):
        # concatenated_row = []
        row = np.array([])
        results = []
        for i in range(20):
            next_row = X[index + i]
            row = np.concatenate((row, next_row))
            if i > 9:
                results.append(y[index + i])
        counts = np.bincount(np.array(results))  # returns result that occurred most in the last 10
        ground_truths.append(np.argmax(counts))
        x_with_20_frames_in_past.append(row)  # combines all inputs of this and the next 20 frames

    X = np.array(x_with_20_frames_in_past)
    y = np.array(ground_truths)
    activation_function = 'sigmoid'
    neural_network = nn.NeuralNetwork((X.shape[1], 20, 20, 1), activation_function=activation_function)
    neural_network.train(X, y, 1000, 0.01)
    predictions = neural_network.forward_pass(X)
    error = np.mean(neural_network.get_error(predictions, y))
    acc = neural_network.accuracy(predictions, y)
    print(error)
    print(acc)
    print(neural_network.weights)
    #save_nn_config(neural_network, acc)
    # np.save(f'weights_{activation_function}_with_{round(acc * 100)}_percent_{datetime.now()}.npy',
    #         neural_network.weights)
    # with open(f'weights_{activation_function}_with_{round(acc * 100)}_percent_accuracy.json', 'w') as file:
    # json.dump(neural_network.weights, f)


def use_existing_config(file_name):
    frames = pd.DataFrame(pd.read_csv(
        "/Users/felixachter/source/uni/machine-learning/neural-network-framework/"
        "data/data_merge/output_csv/rotate_felix_gt.csv"))

    X = frames[get_keys_to_use(frames)].to_numpy()
    y = frames["ground_truth"].to_numpy()
    y[y == "idle"] = 0
    y[y == "rotate"] = 1
    x_with_20_frames_in_past = []
    ground_truths = []
    for index in range(len(X) - 20):
        # concatenated_row = []
        row = np.array([])
        results = []
        for i in range(20):
            next_row = X[index + i]
            row = np.concatenate((row, next_row))
            if i > 9:
                results.append(y[index + i])
        counts = np.bincount(np.array(results))  # returns result that occurred most in the last 10
        ground_truths.append(np.argmax(counts))
        x_with_20_frames_in_past.append(row)  # combines all inputs of this and the next 20 frames

    X = np.array(x_with_20_frames_in_past)
    y = np.array(ground_truths)
    neural_network = load_nn_from_config(file_name)
    predictions = neural_network.forward_pass(X)
    error = np.mean(neural_network.get_error(predictions, y))
    acc = neural_network.accuracy(predictions, y)
    print(error)
    print(acc)





def get_keys_to_use(frames):
    important_keywords = ["shoulder", "elbow", "wrist", "pinky", "index", "thumb"]
    keys_to_use = []
    for key_name in frames.keys():
        if any(include in key_name for include in important_keywords) \
                and "confidence" not in key_name \
                and "foot" not in key_name:
            keys_to_use.append(key_name)
    return keys_to_use


if __name__ == "__main__":
    main()
    # use_existing_config("config_84_accuracy_2022-03-01 13-10-22.json")
