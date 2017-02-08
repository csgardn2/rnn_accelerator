#!/usr/bin/python3.5

import sys
import tensorflow
import random
from tensorflow.python.ops.rnn_cell import LSTMCell

sys.path.append("./training_inputs");

import training_inputs

# This is not called automatically.  It is called by tensorflow.app.run() at
# the bottom of this file.
def main(_):
    
    # Define the network dimensions
    # Feel free to tweak these
    num_batches = 8192
    sequences_per_batch = 32
    time_per_sequence = 16
    training_fraction = 0.8
    
    # Training Fraction:
    # 0 to 1.
    # 0.8 means the first 80% of the training data will be used for training
    # and the last 20% will be used for validation.
    
    # Make sure that any random numbers generated from the python 'random'
    # library (not tensorflow) are always the same between program runs.
    random.seed(42)
    
    if (training_inputs.output_sequence_gold_size != training_inputs.input_sequence_gold_size):
        print(
            "Error.  output_sequence_gold_size is not equal to input_sequence_gold_size",
            file = sys.stderr
        )
        return
    
    # Calculate some additional indexes
    num_training_samples = int(training_inputs.output_sequence_gold_size * training_fraction)
    num_validation_samples = training_inputs.output_sequence_gold_size - num_training_samples
    print("Training range = [0 to {}]".format(num_training_samples - 1))
    print("Validation range = [{} to {}]".format(num_training_samples, training_inputs.output_sequence_gold_size - 1))
    
    # Placeholders will be used to pass arguments to the neural network
    # graphs during training and validation.
    # Naming conventions for placeholders:
    #   <variable name>_<neural | gold>_<tf | native>
    # Dimension ordering:
    #   [sequences_per_batch, time_per_sequence]
    inputs_gold_tf = tensorflow.placeholder(    \
        tensorflow.float32,                     \
        [None, None, 1],                        \
        name = "outputs_gold_tf"                \
    )
    outputs_gold_tf = tensorflow.placeholder(   \
        tensorflow.float32,                     \
        [None, None, 1],                        \
        name = "outputs_gold_tf"                \
    )
    
    # An RNN cell, in the most abstract setting, is anything that has
    # a state and performs some operation that takes a matrix of inputs.
    # This operation results in an output matrix with `self.output_size` columns.
    # If `self.state_size` is an integer, this operation also results in a new
    # state matrix with `self.state_size` columns.  If `self.state_size` is a
    # tuple of integers, then it results in a tuple of `len(state_size)` state
    # matrices, each with a column size corresponding to values in `state_size`.
    cell = LSTMCell(1)
    
    # Unroll the LSTM in time so that training can occur in batches of multiple
    # sequences with multiple time steps.
    unrolled_lstm_neural_tf, state_neural_tf    \
      = tensorflow.nn.dynamic_rnn(              \
        cell,                                   \
        inputs_gold_tf,                         \
        dtype = tensorflow.float32              \
    )
    
    # Trainable variables
    weights_neural_tf = tensorflow.Variable(tensorflow.zeros([1]))
    biases_neural_tf = tensorflow.Variable(tensorflow.zeros([1]))
    
    # Combine LSTM cell with weights and biases
    outputs_neural_tf = tensorflow.add(
        tensorflow.mul(
            unrolled_lstm_neural_tf,
            weights_neural_tf
        ),
        biases_neural_tf
    )
    
    # Define an error function for training and validation
    error_function = tensorflow.reduce_mean(    \
        tensorflow.abs(                         \
            tensorflow.sub(                     \
                outputs_neural_tf,              \
                outputs_gold_tf                 \
            )                                   \
        )                                       \
    )
    optimizer = tensorflow.train.AdamOptimizer(1e0).minimize(error_function)
    
    # Pre-allocate & setup some arrays for training and validation
    max_start_time = num_training_samples - time_per_sequence
    training_inputs_gold_native = sequences_per_batch * [time_per_sequence * [1 * [0.0]]]
    training_outputs_gold_native = sequences_per_batch * [time_per_sequence * [1 * [0.0]]]
    validation_inputs_gold_native = 1 * [num_validation_samples * [1 * [0.0]]]
    validation_outputs_gold_native = 1 * [num_validation_samples * [1 * [0.0]]]
    for time_ix in range(num_validation_samples):
        validation_inputs_gold_native[0][time_ix][0]                            \
          = training_inputs.input_sequence_gold[num_training_samples + time_ix]
        validation_outputs_gold_native[0][time_ix][0]                           \
          = training_inputs.output_sequence_gold[num_training_samples + time_ix]
    
    # These are required to pass arguments between the python runtime
    # and the tensorflow runtime during a call to session.run
    training_argument_map = {                           \
        inputs_gold_tf : training_inputs_gold_native,   \
        outputs_gold_tf : training_outputs_gold_native  \
    }
    validation_argument_map = {                             \
        inputs_gold_tf : validation_inputs_gold_native,     \
        outputs_gold_tf : validation_outputs_gold_native    \
    }
    
    # Prepare runtime to execute training and validation commands
    session = tensorflow.Session()
    session.run(tensorflow.global_variables_initializer())
    
    for batch_ix in range(num_batches):
        
        # Collect training inputs and outputs into a batch
        for sequence_ix in range(sequences_per_batch):
            start_time = int(max_start_time * random.random())
            for time_ix in range(time_per_sequence):
                training_inputs_gold_native[sequence_ix][time_ix][0]            \
                  = training_inputs.input_sequence_gold[start_time + time_ix]
                training_outputs_gold_native[sequence_ix][time_ix][0]           \
                  = training_inputs.output_sequence_gold[start_time + time_ix]
        
        session.run(optimizer, training_argument_map)
        
        # Validation takes a long time, so only evaluate it
        if (batch_ix % 16 == 15):
            validation_error = session.run(error_function, validation_argument_map)
            print("Batch = {} / {} | Validation error = {}".format(batch_ix + 1, num_batches, validation_error))
        

# This is actually the first thing that is executed.  It is hard-coded to
# search for a function called 'main' and call it.
tensorflow.app.run()

