#!/usr/bin/python

import random
import sys
import tensorflow

sys.path.append('./dataset/noise_512_linear_normalized')
sys.path.append('./neural_topologies')

# Load huge python lists of input data
from input_tiles import input_tiles
from output_pixels import output_pixels

from gradient_basic_graph import gradient_basic_graph

def main(_):
    
    # Feel free to tweak these
    num_batches = 65536
    batch_size = 512
    training_fraction = 0.8
    
    num_tiles = len(output_pixels)
    training_bound = int(float(num_tiles) * training_fraction)
    
    # Do some pre-processing on the pixel outputs to convert them to logits
    answers_logits = num_tiles * [256 * [0.0]]
    for ix in range(num_tiles):
        lard = int(256 * output_pixels[ix]) % 256
        poop = 256 * [0.0]
        poop[lard] = 1.0;
        answers_logits[ix] = poop
    
    # Define the neural network topology
    neural_inputs_1 = tensorflow.placeholder(tensorflow.float32, [None, 9])
    answers = tensorflow.placeholder(tensorflow.int64, [None, 256])
    
#    base_signals_2 = tensorflow.reduce_min(neural_inputs_1, reduction_indices=[1])
    
#    clipped_inputs_3 = tensorflow.transpose(
#        tensorflow.sub(
#            tensorflow.transpose(neural_inputs_1),
#            tensorflow.transpose(base_signals_2)
#        )
#    )
    # Clipping turns out to be unnecessary, which is actually a good thing since
    # it means that a general purpose net is just as useful as one where we
    # cheat a bit.
    clipped_inputs_3 = neural_inputs_1
    
    neural_weights_3 = tensorflow.Variable(tensorflow.zeros([9, 256]))
    neural_biases_3 = tensorflow.Variable(tensorflow.zeros([1, 256]))
    
    neural_outputs_4 = tensorflow.add(
        tensorflow.matmul(clipped_inputs_3, neural_weights_3),
        neural_biases_3
    )
    
    # Define an error function to be minimised
    error_function = tensorflow.reduce_mean(
        tensorflow.nn.softmax_cross_entropy_with_logits(
            neural_outputs_4,
            answers
        )
    )
    train_step = tensorflow.train.AdamOptimizer(1e-4).minimize(error_function)
    
    # Fill variables with zeros
    session = tensorflow.InteractiveSession()
    tensorflow.initialize_all_variables().run()
    
    # Train the neural network
    batch_inputs = batch_size * [9 * [0]]
    batch_answers = batch_size * [256 * [0]]
    sample_indexes = batch_size * [[0]]
    for iy in range(num_batches):
        
        sample_indexes = random.sample(range(training_bound), batch_size)
        for ix in range(batch_size):
            batch_inputs[ix] = input_tiles[sample_indexes[ix]]
            batch_answers[ix] = answers_logits[sample_indexes[ix]]
        
        input_bindings = {
            neural_inputs_1 : batch_inputs,
            answers : batch_answers
        }
        
        session.run(train_step, input_bindings)
    
    # Validate the neural net
    
    validation_function = tensorflow.reduce_mean(
        tensorflow.cast(
            tensorflow.abs(
                tensorflow.sub(
                    tensorflow.argmax(neural_outputs_4, 1),
                    tensorflow.argmax(answers, 1)
                )
            ),
            tensorflow.float32
        )
    )
    
    input_bindings = {
        neural_inputs_1 : input_tiles[training_bound:],
        answers : answers_logits[training_bound:]
    }
    
    session.run(
        tensorflow.Print(
            neural_biases_3,
            [neural_biases_3],
            summarize = 1024 * 1024 * 1024
        )
    )
    session.run(
        tensorflow.Print(
            neural_weights_3,
            [neural_weights_3],
            summarize = 1024 * 1024 * 1024
        )
    )
    
    print 'Neural net accuracy = {0:.2f}%'.format(
        100.0 * (1.0 - (session.run(validation_function, input_bindings)) / 256.0)
    )
    
    

# Actually run the network and call main() above, up until now, weve only queued
# up work to do
tensorflow.app.run()

