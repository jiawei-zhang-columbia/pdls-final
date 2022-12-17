import time
import tensorflow as tf


class timecallback(tf.keras.callbacks.Callback):
    """
    Custom callback to record wall clock time
    """
    def __init__(self):
        self.times = []
        # use this value as reference to calculate cummulative time taken
        self.timetaken = time.process_time()
    def on_epoch_end(self,epoch, logs = {}):
        self.times.append((epoch, time.process_time() - self.timetaken))

