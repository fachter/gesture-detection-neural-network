from math import ceil

import pandas as pd
import numpy as np
from neural_net.data_preprocessor import DataPreprocessor
from nn_framework_package.nn_session_util import load_from_config
import yaml
import argparse


def run_test_mode(input_filename, output_filename):
    included_frames = 20
    first_range = ceil(0.7 * included_frames)
    data_processor = DataPreprocessor(including_frames=included_frames)
    nn, pca = load_from_config("../neural_net/saved_configs/mandatory/rel_70/config_85_f1_2022-04-01 22-24-03.pkl")[0:2]
    cols = get_frame_columns()
    frames = pd.DataFrame(columns=cols)
    print("Reading in: ", input_filename)
    csv = pd.read_csv(input_filename)
    frames = frames.append(csv, ignore_index=True)
    input_values = nn.scaler.transform(pca.transform_data(data_processor.preprocess_data(frames)))

    predictions = nn.predict(input_values)
    results = ["idle"]
    for _ in range(first_range):
        results.append("idle")
    previous_prediction = "idle"
    previous_saved_prediction = "idle"
    prediction_count = 0
    for prediction in predictions:
        if prediction == previous_prediction:
            prediction_count += 1
        else:
            prediction_count = 1

        if prediction_count >= 4 and previous_saved_prediction != prediction:
            previous_saved_prediction = prediction
            results.append(prediction)
        else:
            results.append("idle")
        previous_prediction = prediction
    for _ in range(included_frames - 1 - first_range):
        results.append("idle")
    pd.DataFrame(np.c_[frames["timestamp"], results], columns=["timestamp", "events"]).to_csv(output_filename)
    print("Saved to: ", output_filename)


def get_frame_columns():
    with open("keypoint_mapping.yml", "r") as yaml_file:
        mappings = yaml.safe_load(yaml_file)
        keypoint_names = mappings["face"]
        keypoint_names += mappings["body"]
    cols = ["timestamp"]
    for keypoint_name in keypoint_names:
        cols.append(keypoint_name + "_x")
        cols.append(keypoint_name + "_y")
        cols.append(keypoint_name + "_z")
        cols.append(keypoint_name + "_visibility")
    return cols


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", help="CSV file containing the video transcription from OpenPose",
                        default="video_frames_input.csv")
    parser.add_argument("--output_csv", help="CSV file containing the video transcription from OpenPose",
                        default="performance_results.csv")

    args = parser.parse_known_args()[0]
    input_csv = args.input_csv
    output_csv = args.output_csv
    run_test_mode(input_csv, output_csv)
