import os
import time

import cv2
import mediapipe as mp
import yaml


# This script uses video_capturer to parse videos to extract coordinates of
# the user's joints. You find documentation about video_capturer here:
#  https://google.github.io/mediapipe/solutions/pose.html
class VideoCapturer:
    data = []

    def run(self):
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_pose = mp.solutions.pose

        # ===========================================================
        # ======================= SETTINGS ==========================
        show_video = True
        show_data = True

        # Live from camera (change index if you have more than one camera)
        cap = cv2.VideoCapture(index=1)

        # ===========================================================

        # the names of each joint ("keypoint") are defined in this yaml file:
        print(os.listdir())
        with open("keypoint_mapping.yml", "r") as yaml_file:
            mappings = yaml.safe_load(yaml_file)
            KEYPOINT_NAMES = mappings["face"]
            KEYPOINT_NAMES += mappings["body"]
        success = True
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened() and success:
                success, image = cap.read()
                if not success:
                    break
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                print(results)
                time.sleep(2)

                # Draw the pose annotation on the image
                if show_video:
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                              landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                    cv2.imshow('MediaPipe Pose', image)

                if cv2.waitKey(5) & 0xFF == 27:
                    break

                # =================================
                # ===== read and process data =====
                if show_data and results.pose_landmarks is not None:
                    result = f"timestamp: {str(cap.get(cv2.CAP_PROP_POS_MSEC))}  "
                    for i in range(32):
                        result += f"{KEYPOINT_NAMES[i]}_x: {str(results.pose_landmarks.landmark[i].x)}  "
                        result += f"{KEYPOINT_NAMES[i]}_y: {str(results.pose_landmarks.landmark[i].y)}  "
                        result += f"{KEYPOINT_NAMES[i]}_z: {str(results.pose_landmarks.landmark[i].z)}  "
                        result += f"{KEYPOINT_NAMES[i]}_visibility: {str(results.pose_landmarks.landmark[i].visibility)}  "
                    self.data.append(result)
                if len(self.data) > 30:
                    del self.data[0]
                # ==================================
        cap.release()
