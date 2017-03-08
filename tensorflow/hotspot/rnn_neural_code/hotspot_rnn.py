#!/usr/bin/python3.5

import sys
import tensorflow
import random
import load_training_data
from tensorflow.python.ops.rnn_cell import LSTMCell
from load_training_data import load_training_data

def main(_):
    
    load_training_data(                                             \
        "training_data/power_256x256_full_dynamicpower_iteration0", \
        ".csv",                                                     \
        250,                                                        \
        6,                                                          \
        256,                                                        \
        256                                                         \
    )
    

# This is actually the first thing that is executed.  It is hard-coded to
# search for a function called 'main' and call it.
tensorflow.app.run()


