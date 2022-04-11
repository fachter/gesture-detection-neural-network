import os
import argparse
import pandas as pd
import random

parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", help="CSV file containing the video transcription from OpenPose", required=True)
parser.add_argument("--output_csv_name", help="output CSV file containing the events", default="emitted_events.csv")

args = parser.parse_known_args()[0]

input_path = args.input_csv

output_directory, input_csv_filename = os.path.split(args.input_csv)
output_path = "%s/%s" % (output_directory, args.output_csv_name)

frames = pd.read_csv(input_path, index_col="timestamp")
frames.index = frames.index.astype(int)

# ================================= your application =============================
# you should import and call your own application here

class DemoApplication():
  available_gestures = ["idle", "swipe_left", "swipe_right", "rotate"]
  weights =            [ 0.97,   0.01,         0.01,          0.01]

  # make sure you simulate live prediction; this means that for each frame you must only regard the data of the
  # current frame or past frames, never future frames!
  def compute_events(self, frames):
    return random.choices(self.available_gestures, weights=self.weights, k=len(frames)) # returns a random event for each frame


my_model = DemoApplication()
# ================================================================================

# determine events
frames["events"] = my_model.compute_events(frames)

# the CSV has to have the columns "timestamp" and "events"
# but may also contain additional columns, which will be ignored during the score evaluation
frames["events"].to_csv(output_path, index=True) # since "timestamp" is the index, it will be saved also
print("events exported to %s" % output_path)
