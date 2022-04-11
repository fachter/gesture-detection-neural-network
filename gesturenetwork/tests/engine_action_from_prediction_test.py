import unittest
from datetime import datetime
import datetime as dt

import numpy as np

from ..app_engine.app_engine import AppEngine
from multiprocessing import Pipe, Process


def return_same(x):
    return x


class MyTestCase(unittest.TestCase):
    def test_same_gesture_with_high_confidence_then_send_action(self):
        engine = AppEngine(action_handler_function=return_same,
                           config_filename="../neural_net/saved_configs/mandatory/"
                                           "rel_70/config_85_f1_2022-04-01 22-24-03.pkl")
        prediction_receiver, prediction_sender = Pipe(duplex=False)
        action_receiver, action_sender = Pipe(duplex=False)
        
        pred = np.array([["swipe_left", 0.999999]])
        prediction_sender.send(pred)
        prediction_sender.send(pred)
        prediction_sender.send(pred)
        p = Process(target=engine.action_from_prediction,
                    args=(prediction_receiver, action_sender))
        p.daemon = True
        p.start()

        actions = []
        timer = datetime.now() + dt.timedelta(seconds=1)
        while timer > datetime.now():
            if action_receiver.poll():
                actions.append(action_receiver.recv())

        self.assertEqual(1, len(actions))
        self.assertEqual("swipe_left", actions[-1])

        p.terminate()

    def test_same_gesture_with_low_confidence_then_do_not_action(self):
        engine = AppEngine(action_handler_function=return_same,
                           config_filename="../neural_net/saved_configs/mandatory/"
                                           "rel_70/config_85_f1_2022-04-01 22-24-03.pkl")
        prediction_receiver, prediction_sender = Pipe(duplex=False)
        action_receiver, action_sender = Pipe(duplex=False)

        pred = np.array([["swipe_left", 0.5]])
        prediction_sender.send(pred)
        prediction_sender.send(pred)
        prediction_sender.send(pred)
        p = Process(target=engine.action_from_prediction,
                    args=(prediction_receiver, action_sender))
        p.daemon = True
        p.start()

        actions = []
        timer = datetime.now() + dt.timedelta(seconds=1)
        while timer > datetime.now():
            if action_receiver.poll():
                actions.append(action_receiver.recv())

        self.assertEqual(0, len(actions))

        p.terminate()

    def test_same_gesture_with_idle_in_between_then_send_twice(self):
        engine = AppEngine(action_handler_function=return_same,
                           config_filename="../neural_net/saved_configs/mandatory/"
                                           "rel_70/config_85_f1_2022-04-01 22-24-03.pkl")
        prediction_receiver, prediction_sender = Pipe(duplex=False)
        action_receiver, action_sender = Pipe(duplex=False)

        pred = np.array([["rotate", 0.9999]])
        idle = np.array([["idle", 0.9999]])

        prediction_sender.send(pred)
        prediction_sender.send(pred)
        prediction_sender.send(idle)
        prediction_sender.send(idle)
        prediction_sender.send(pred)
        prediction_sender.send(pred)
        p = Process(target=engine.action_from_prediction,
                    args=(prediction_receiver, action_sender))
        p.daemon = True
        p.start()

        actions = []
        timer = datetime.now() + dt.timedelta(seconds=2)
        while timer > datetime.now():
            if action_receiver.poll():
                actions.append(action_receiver.recv())

        self.assertEqual(3, len(actions))
        self.assertEqual("rotate", actions[0])
        self.assertEqual("idle", actions[1])
        self.assertEqual("rotate", actions[2])

        p.terminate()

    def test_gesture_with_noice_in_between_then_send_gesture(self):
        engine = AppEngine(action_handler_function=return_same,
                           config_filename="../neural_net/saved_configs/mandatory/"
                                           "rel_70/config_85_f1_2022-04-01 22-24-03.pkl")
        prediction_receiver, prediction_sender = Pipe(duplex=False)
        action_receiver, action_sender = Pipe(duplex=False)

        swipe_right = np.array([["swipe_right", 0.9999]])
        flip_table = np.array([["flip_table", 0.7]])

        prediction_sender.send(swipe_right)
        prediction_sender.send(flip_table)
        prediction_sender.send(swipe_right)
        p = Process(target=engine.action_from_prediction,
                    args=(prediction_receiver, action_sender))
        p.daemon = True
        p.start()

        actions = []
        timer = datetime.now() + dt.timedelta(seconds=1)
        while timer > datetime.now():
            if action_receiver.poll():
                actions.append(action_receiver.recv())

        self.assertEqual(1, len(actions))
        self.assertEqual("swipe_right", actions[0])

        p.terminate()

    def test_longer_run(self):
        engine = AppEngine(action_handler_function=return_same,
                           config_filename="../neural_net/saved_configs/mandatory/"
                                           "rel_70/config_85_f1_2022-04-01 22-24-03.pkl")
        prediction_receiver, prediction_sender = Pipe(duplex=False)
        action_receiver, action_sender = Pipe(duplex=False)

        rotate_right = np.array([["rotate_right", 0.9999]])
        rotate_left = np.array([["rotate_left", 0.9999]])
        idle = np.array([["idle", 0.9999]])

        prediction_sender.send(idle)
        prediction_sender.send(idle)
        prediction_sender.send(rotate_left)
        prediction_sender.send(idle)
        prediction_sender.send(idle)
        prediction_sender.send(rotate_right)
        prediction_sender.send(idle)
        prediction_sender.send(idle)
        prediction_sender.send(rotate_left)
        prediction_sender.send(rotate_right)
        prediction_sender.send(rotate_right)
        p = Process(target=engine.action_from_prediction,
                    args=(prediction_receiver, action_sender))
        p.daemon = True
        p.start()

        actions = []
        timer = datetime.now() + dt.timedelta(seconds=1)
        while timer > datetime.now():
            if action_receiver.poll():
                actions.append(action_receiver.recv())

        self.assertEqual(2, len(actions))
        self.assertEqual("idle", actions[0])
        self.assertEqual("rotate_right", actions[1])

        p.terminate()


if __name__ == '__main__':
    unittest.main()
