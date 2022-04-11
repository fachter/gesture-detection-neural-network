import argparse


def get_camera_index_from_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_index", help="Index of the camera to use for detecting gestures",
                        default=0)
    return int(parser.parse_known_args()[0].camera_index)
