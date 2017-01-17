#!/usr/bin/python

import random
import sys
import tensorflow

sys.path.append('./dataset/noise_512_linear_nonnormalized');
sys.path.append('./neural_topologies');

from input_tiles import input_tiles
from output_pixels import output_pixels
from gradient_basic_graph import gradient_basic_graph

def main(_):
    
    # True - Load weights and biases from a file
    # False - Retrain, discard old data, and overwrite training files if successful
    load_training_data = True
    training_file = './training_results/gradient_basic_graph/variables.bin'
    
    # Feel free to tweak these
    num_batches = 65536
    batch_size = 512
    training_fraction = 0.8 # (floating point 0 to 1)
    
    # Tensorflow likes matricies with one of the demensions being 1 better
    # than a straight up 1D list.  Re-structure the 1D answers list
    num_tiles = len(input_tiles)
    mat_output_pixels = num_tiles * [[0]]
    for iy in range(num_tiles):
        mat_output_pixels[iy] = [output_pixels[iy]]
    
    # Build the entire neural network topology which takes a single 28 x 28
    # image as an input and produces 10 probabilities for each digit as an
    # output.  Does not actually initiate training yet.
    neural_inputs = tensorflow.placeholder(tensorflow.float32, [None, 9])
    
    # neural_outputs = gradient_basic_graph(neural_inputs)
    neural_weights = tensorflow.Variable(tensorflow.zeros([9, 1]))
    neural_biases = tensorflow.Variable(tensorflow.zeros([1]))
    neural_outputs = tensorflow.add(
        tensorflow.matmul(
            neural_inputs,
            neural_weights
        ),
        neural_biases
    )
    
    answers = tensorflow.placeholder(tensorflow.float32, [None, 1])
    
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
    train_step = tensorflow.train.AdamOptimizer(1e-2).minimize(mean_error)
    
    # Execute statements such as tensorflow.truncated_normal and
    # tensorflow.zeros so that all neural layers start with some value before
    # training begins
    session = tensorflow.InteractiveSession()
    tensorflow.initialize_all_variables().run()
    
    training_bound = int(training_fraction * num_tiles)
    print 'Validation Range =', training_bound, 'to', num_tiles - 1
    
    if load_training_data:
        
        # Load pre-existing trained weights and biases from a file
        saver = tensorflow.train.Saver()
        saver.restore(session, training_file)
        
    else:
        
        # Perform training now and save the resultant weights and biases to a file
        
        print 'Training Range = 0 to', training_bound - 1
        
        # Train the network using many small batches (using all the training data
        # at once would be too computationally intensive).
        batch_inputs = batch_size * [9 * [0]]
        batch_answers = batch_size * [[0]]
        
        for ix in range(num_batches):
            
            # Select a small subset of the training data
            sample_indexes = random.sample(range(training_bound), batch_size)
            for iy in range(batch_size):
                
                cur_index = sample_indexes[iy]
                batch_inputs[iy] = input_tiles[cur_index]
                batch_answers[iy] = mat_output_pixels[cur_index] # labels
            
            # This will train the neural net using only data in the batch.  Once
            # this small training step is done, the resultant weights and biases
            # are merged with the weights and biases of all previous training steps
            batch_data = {neural_inputs : batch_inputs, answers : batch_answers}
            session.run(train_step, batch_data)
        
        # Write the trained weights and biases to a file
        saver = tensorflow.train.Saver()
        save_path = saver.save(session, training_file)
        print 'Model saved in file', save_path
        
    
    # Validate the trained network
    comparison = tensorflow.abs(tensorflow.sub(neural_outputs, answers))
    accuracy = tensorflow.reduce_mean(tensorflow.cast(comparison, tensorflow.float32))
    
    print session.run(neural_weights)
    print session.run(neural_biases)
    
    input_bindings = {
        neural_inputs : input_tiles[training_bound : num_tiles],
        answers : mat_output_pixels[training_bound : num_tiles]
    }
    print 'Neural net accuracy = {0:.2f}%'.format(
        100.0 / 256.0 * (256.0 - session.run(accuracy, input_bindings))
    )
    

# Actually run the network and call main() above, up until now, weve only queued
# up work to do
tensorflow.app.run()

