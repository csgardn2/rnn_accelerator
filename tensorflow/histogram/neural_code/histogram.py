#!/usr/bin/python

import random
import sys
import tensorflow

sys.path.append('./dataset');
sys.path.append('./neural_topologies');

from input_samples import input_samples
from output_histograms import output_histograms
from histogram_basic_graph import histogram_basic_graph
from histogram_four_layer_graph import histogram_four_layer_graph
from histogram_relu_graph import histogram_relu_graph
from histogram_softsign_graph import histogram_softsign_graph

def main(_):
    
    # True - Load weights and biases from a file
    # False - Retrain, discard old data, and overwrite training files if successful
    load_training_data = False
    training_file = './training_results/histogram_basic_graph/variables.bin'
    
    # Feel free to tweak these
    num_batches = 65536
    batch_size = 512
    training_fraction = 0.8 # (floating point 0 to 1)
    
    # Tensorflow likes matricies with one of the demensions being 1 better
    # than a straight up 1D list.  Re-structure the 1D answers list
    num_samples = len(input_samples)
    inputs_per_sample = len(input_samples[0])
    num_histogram_bins = len(output_histograms[0])
    
    # Build the entire neural network topology which takes a single 28 x 28
    # image as an input and produces 10 probabilities for each digit as an
    # output.  Does not actually initiate training yet.
    # Tensorflow conventions
    # [height, width]
    neural_inputs = tensorflow.placeholder(
        tensorflow.float32,
        [None, inputs_per_sample]
    )
    
    neural_outputs = histogram_softsign_graph(
        neural_inputs,
        inputs_per_sample,
        num_histogram_bins
    )
    
    answers = tensorflow.placeholder(tensorflow.float32, [None, num_histogram_bins])
    
    # Define an error function and optimization method to measure how well our
    # neural net does during training.
    mean_error = tensorflow.reduce_mean(
        tensorflow.abs(
            tensorflow.sub(
                neural_outputs,
                answers
            )
        )
    )
    min_error = tensorflow.reduce_min(
        tensorflow.abs(
            tensorflow.sub(
                neural_outputs,
                answers
            )
        )
    )
    max_error = tensorflow.reduce_max(
        tensorflow.abs(
            tensorflow.sub(
                neural_outputs,
                answers
            )
        )
    )    
    train_step = tensorflow.train.AdamOptimizer(1e-2).minimize(mean_error)
    
    # Execute statements such as tensorflow.truncated_normal and
    # tensorflow.zeros so that all neural layers start with some value before
    # training begins
    session = tensorflow.InteractiveSession()
    tensorflow.initialize_all_variables().run()
    
    training_bound = int(training_fraction * num_samples)
    print 'Training Range = 0 to', training_bound - 1
    print 'Validation Range =', training_bound, 'to', num_samples - 1
    
    # Train the network using many small batches (using all the training data
    # at once would be too computationally intensive).
    batch_inputs = batch_size * [inputs_per_sample * [0]]
    batch_answers = batch_size * [num_histogram_bins * [0]]
    for ix in range(num_batches):
        
        # Select a small subset of the training data
        sample_indexes = random.sample(range(training_bound), batch_size)
        for iy in range(batch_size):
            
            cur_index = sample_indexes[iy]
            batch_inputs[iy] = input_samples[cur_index]
            batch_answers[iy] = output_histograms[cur_index] # labels
        
        # This will train the neural net using only data in the batch.  Once
        # this small training step is done, the resultant weights and biases
        # are merged with the weights and biases of all previous training steps
        batch_data = {neural_inputs : batch_inputs, answers : batch_answers}
        session.run(train_step, batch_data)
    
    # Validate the neural network
    batch_data = {
        neural_inputs : input_samples[training_bound:],
        answers : output_histograms[training_bound:]
    }
    
    reduced_error = session.run(mean_error, batch_data)
    
    histogram_of_naughtyness = tensorflow.histogram_fixed_width(
        tensorflow.abs(
            tensorflow.sub(
                neural_outputs,
                answers
            )
        ),
        [0.0, 32.0],
        32
    )
    print session.run(histogram_of_naughtyness, batch_data)
    
    print 'inputs_per_sample = {}'.format(inputs_per_sample)
    print 'num_histogram_bins = {}'.format(num_histogram_bins)
    print 'mean error = {0:.2f}'.format(reduced_error)
    print 'min error = {0:.2f}'.format(session.run(min_error, batch_data))
    print 'max error = {0:.2f}'.format(session.run(max_error, batch_data))
    print 'neural net accuracy = {0:.2f}%'.format(100.0 - 100.0 * reduced_error / float(num_histogram_bins))
    
    

# Actually run the network and call main() above, up until now, weve only queued
# up work to do
tensorflow.app.run()

