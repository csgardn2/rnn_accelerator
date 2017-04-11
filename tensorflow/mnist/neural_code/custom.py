import random
import sys
import tensorflow

sys.path.append('./dataset');
sys.path.append('./neural_topologies');

from mnist_images_1d import mnist_images
from mnist_labels import mnist_labels
from mnist_multilayer_graph import mnist_multilayer_graph
from mnist_basic_graph import mnist_basic_graph

def main(_):
    
    # True - Load weights and biases from a file
    # False - Retrain, discard old data, and overwrite training files if successful
    load_training_data = True
    training_file = "./training_results/mnist_multilayer_graph/variables.bin"
    
    # Feel free to tweak these
    num_batches = 65536
    batch_size = 256
    training_fraction = 0.8 # (floating point 0 to 1)
    
    # My machine has 2 gpus - use the auxillary one and leave gpu0 for rendering
    # the desktop / GPU.
    # with tensorflow.device('/gpu:1'):
    
    # Build the entire neural network topology which takes a single 28 x 28
    # image as an input and produces 10 probabilities for each digit as an
    # output.  Does not actually initiate training yet.
    neural_inputs = tensorflow.placeholder(tensorflow.float32, [None, 784])
    neural_outputs = mnist_multilayer_graph(neural_inputs)
    answers = tensorflow.placeholder(tensorflow.float32, [None, 10])
    return;
    
    # Define an error function and optimization method to measure how well our
    # neural net does during training.
    cross_entropy = tensorflow.reduce_mean(tensorflow.nn.softmax_cross_entropy_with_logits(neural_outputs, answers))
    train_step = tensorflow.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    
    # Execute statements such as tensorflow.truncated_normal and
    # tensorflow.zeros so that all neural layers start with some value before
    # training begins
    session = tensorflow.InteractiveSession()
    tensorflow.initialize_all_variables().run()
    
    num_images = len(mnist_images)
    training_bound = int(training_fraction * num_images)
    print 'Validation Range =', training_bound, 'to', num_images - 1
    
    # Convert the 1D array of labels to a 2D one-hot representation
    one_hot_labels = num_images * [10 * [0]]
    for ix in range(0, num_images):
        one_hot_labels[ix] = [int(mnist_labels[ix] == digit) for digit in range(0, 10)]
    
    if load_training_data:
        
        # Load pre-existing trained weights and biases from a file
        saver = tensorflow.train.Saver()
        saver.restore(session, training_file)
        
    else:
        
        # Perform training now and save the resultant weights and biases to a file
        
        print 'Training Range = 0 to', training_bound - 1
        
        # Train the network using many small batches (using all the training data
        # at once would be too computationally intensive).
        batch_inputs = batch_size * [784 * [0]]
        batch_answers = batch_size * [10 * [0]]
        for ix in range(num_batches):
            
            # Select a small subset of the training data
            sample_indexes = random.sample(range(training_bound), batch_size)
            for iy in range(batch_size):
                batch_inputs[iy] = mnist_images[sample_indexes[iy]]
                batch_answers[iy] = one_hot_labels[sample_indexes[iy]]
            
            # This will train the neural net using only data in the batch.  Once
            # this small training step is done, the resultant weights and biases
            # are merged with the weights and biases of all previous training steps
            batch_data = {neural_inputs : batch_inputs, answers : batch_answers}
            session.run(train_step, batch_data)
        
        # Write the trained weights and biases to a file
        saver = tensorflow.train.Saver()
        save_path = saver.save(session, training_file)
        print "Model saved in file", save_path
        
    
    # Validate the trained network
    comparison = tensorflow.equal(tensorflow.argmax(neural_outputs, 1), tensorflow.argmax(answers, 1))
    accuracy = tensorflow.reduce_mean(tensorflow.cast(comparison, tensorflow.float32))
    print(
        session.run(
            accuracy, {
                neural_inputs : mnist_images[training_bound : num_images],
                answers : one_hot_labels[training_bound : num_images]
            }
        )
    )
    
 
# Actually run the network and call main() above, up until now, weve only queued
# up work to do
tensorflow.app.run()

