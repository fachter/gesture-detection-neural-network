import asyncio
import time
import sys
from datetime import datetime
import datetime as dt
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
import mediapipe as mp
import numpy as np

import pandas as pd
import yaml

import gesturenetwork.neural_net.session_util as session_util
from netneural.session.nn_session_util import load_from_config
from gesturenetwork.neural_net.data_preprocessor import DataPreprocessor

from cv2 import cv2

np.set_printoptions(threshold=sys.maxsize)


class AppEngine:
    def __init__(self, config_filename, action_handler_function, old=False, camera_index=0):
        self.nn, self.pca = load_from_config(config_filename)[:2]

        frames, _, _, X_train = session_util.load_session_from_config(config_filename)[0:4]

        if old:
            # set scaler of nn (for old configs)
            self.nn.scaler.fit(X_train)  # TODO: scaler never set for new configs?

        self.action_handler = action_handler_function
        self.should_run = False
        self.preprocessing_including_frames = frames
        self.action_receiver: Connection | None = None
        self.camera_index = camera_index

    def process_image_to_frame_row(self, row_sender: Connection, should_run: Connection):
        video_cap = None
        try:
            drawing = mp.solutions.drawing_utils
            drawing_styles = mp.solutions.drawing_styles
            mp_pose = mp.solutions.pose

            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0) as pose:
                video_cap = cv2.VideoCapture(index=self.camera_index)
                while not video_cap.isOpened():
                    time.sleep(0.01)
                while video_cap.isOpened():
                    success, image = video_cap.read()
                    img = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                    img.flags.writeable = False
                    result = pose.process(img)
                    landmarks = result.pose_landmarks
                    if landmarks is not None:
                        img.flags.writeable = True
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                        drawing.draw_landmarks(img, landmarks, mp_pose.POSE_CONNECTIONS,
                                               landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style())
                        cv2.imshow("Perform Gestures!", img)
                        if cv2.waitKey(5) & 0xFF == 27 and self.should_run:
                            should_run.send(False)
                        if self.should_run:
                            row = [datetime.now()]
                            for item in landmarks.landmark:
                                row.append(item.x)
                                row.append(item.y)
                                row.append(item.z)
                                row.append(item.visibility)
                            row_sender.send(row)
                        else:
                            if should_run.poll():
                                self.should_run = should_run.recv()

        except KeyboardInterrupt:
            if video_cap is not None:
                video_cap.release()
            cv2.destroyAllWindows()
            print("\nShutdown completed\n")

    def preprocess_frames(self, row_receiver: Connection, preprocessed_data_sender: Connection):
        preprocessor = DataPreprocessor(self.preprocessing_including_frames)
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
        frames = pd.DataFrame(columns=cols)
        resampled_frames = pd.DataFrame(columns=cols)

        included_frames = self.preprocessing_including_frames
        fps = 30
        first_resample = (included_frames / fps) * dt.timedelta(seconds=1)

        try:
            while True:
                if row_receiver.poll():
                    row = row_receiver.recv()
                    frames.loc[len(frames)] = row
                    if (frames['timestamp'][len(frames) - 1] - frames['timestamp'][0]) > first_resample:
                        resampled_frames = frames.set_index('timestamp').resample('25ms', origin='start')\
                            .max().interpolate('linear')
                        process_and_predict = True
                    else:
                        process_and_predict = False

                    if process_and_predict:
                        data_in = preprocessor.preprocess_data(resampled_frames[:included_frames + 1])
                        processed_data = self.pca.transform_data(data_in)
                        preprocessed_data_sender.send(processed_data)
                        indicis_to_drop = frames[frames['timestamp'] <
                                                 (datetime.now() - first_resample)].index.values
                        frames.drop(indicis_to_drop, inplace=True)
                        frames.reset_index(drop=True, inplace=True)
        except KeyboardInterrupt:
            pass

    def predict(self, preprocessed_data: Connection, predictions: Connection):
        try:
            while True:
                if preprocessed_data.poll():
                    input_data = self.nn.scaler.transform(preprocessed_data.recv())
                    prediction = self.nn.predict(input_data, True)
                    predictions.send(prediction)
        except KeyboardInterrupt:
            pass

    def action_from_prediction(self, prediction_receiver: Connection, action_sender: Connection):
        previous_predictions = dict()
        previous_saved_pred = None
        try:
            while True:
                if prediction_receiver.poll():
                    preds_with_confidence = prediction_receiver.recv()
                    confidence = float(preds_with_confidence[:, 1][-1])
                    pred = preds_with_confidence[:, 0][-1]
                    if pred in previous_predictions.keys():
                        previous_predictions[pred] += confidence
                    else:
                        previous_predictions[pred] = confidence
                    if previous_predictions[pred] >= 1.8 and pred != previous_saved_pred:
                        previous_predictions = dict()
                        previous_saved_pred = pred
                        action = self.action_handler(pred)
                        if action is not None:
                            action_sender.send(action)
                    if "idle" in previous_predictions.keys() and previous_predictions["idle"] > 1.8:
                        previous_predictions = dict()
                        previous_saved_pred = "idle"
        except KeyboardInterrupt:
            pass

    async def run(self):
        row_receiver, row_sender = Pipe(duplex=False)
        preprocessed_data_receiver, preprocessed_data_sender = Pipe(duplex=False)
        prediction_receiver, prediction_sender = Pipe(duplex=False)
        self.action_receiver, action_sender = Pipe(duplex=False)
        should_run_main, should_run_process = Pipe()
        await asyncio.sleep(0.001)

        image_to_frame_process = Process(target=self.process_image_to_frame_row,
                                         args=(row_sender, should_run_process))
        network_preprocessing_process = Process(target=self.preprocess_frames,
                                                args=(row_receiver, preprocessed_data_sender))
        network_prediction_process = Process(target=self.predict,
                                             args=(preprocessed_data_receiver, prediction_sender))
        action_process = Process(target=self.action_from_prediction,
                                 args=(prediction_receiver, action_sender))

        processed = [image_to_frame_process, network_preprocessing_process,
                     network_prediction_process, action_process]
        for process in processed:
            process.daemon = True
            process.start()

        print("\n\nEngine booted and ready to use!\n\n")
        while not self.should_run:
            await asyncio.sleep(0.001)
        should_run_main.send(True)
        print("\n\nAccepting gestures now!\n\n")
        while self.should_run:
            if should_run_main.poll():
                self.should_run = should_run_main.recv()
            await asyncio.sleep(0.001)

    async def emit_events(self, ws):
        self.should_run = True
        try:
            while True:
                if self.action_receiver is not None and self.action_receiver.poll():
                    action = self.action_receiver.recv()
                    print("Action: ", action)
                    await ws.send(action)
                await asyncio.sleep(0.001)
        except KeyboardInterrupt:
            pass
