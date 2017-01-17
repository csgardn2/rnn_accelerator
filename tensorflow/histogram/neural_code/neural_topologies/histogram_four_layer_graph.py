import sys
import tensorflow

# Note:
#   Variables declared inside a function are local, and are destroyed when
#   the function returns.  However, the item returned by tensorflow.Variable
#   and tensorflow.placeholder are meerly handles to objects declared inside
#   of tensorflow's runtime environment.  The neural topology persists outside
#   of this function even when the handle goes out of scope and is destroyed.
#
# Best accuracy = 
def histogram_four_layer_graph(neural_inputs, inputs_per_sample, num_histogram_bins):
    
    # Tensorflow conventions
    # [height, width]
    
    # Layer 1
    # Input dimensions = [None, inputs_per_sample]
    # Output dimesnions = [None, 1024]
    neural_weights_1 = tensorflow.Variable(tensorflow.zeros([inputs_per_sample, 1024]))
    neural_biases_1 = tensorflow.Variable(tensorflow.zeros([1, 1024]))
    neural_outputs_1 = tensorflow.add(
        tensorflow.matmul(
            neural_inputs,
            neural_weights_1
        ),
        neural_biases_1
    )
    
    # Layer 2
    # Input dimensions = [None, 1024]
    # Output dimesnions = [None, 512]
    neural_weights_2 = tensorflow.Variable(tensorflow.zeros([1024, 512]))
    neural_biases_2 = tensorflow.Variable(tensorflow.zeros([1, 512]))
    neural_outputs_2 = tensorflow.add(
        tensorflow.matmul(
            neural_outputs_1,
            neural_weights_2
        ),
        neural_biases_2
    )
    
    # Layer 3
    # Input dimensions = [None, 512]
    # Output dimesnions = [None, 256]
    neural_weights_3 = tensorflow.Variable(tensorflow.zeros([512, 256]))
    neural_biases_3 = tensorflow.Variable(tensorflow.zeros([1, 256]))
    neural_outputs_3 = tensorflow.add(
        tensorflow.matmul(
            neural_outputs_2,
            neural_weights_3
        ),
        neural_biases_3
    )
    
    # Layer 3
    # Input dimensions = [None, 256]
    # Output dimesnions = [None, num_histogram_bins]
    neural_weights_4 = tensorflow.Variable(tensorflow.zeros([256, num_histogram_bins]))
    neural_biases_4 = tensorflow.Variable(tensorflow.zeros([1, num_histogram_bins]))
    neural_outputs_4 = tensorflow.add(
        tensorflow.matmul(
            neural_outputs_3,
            neural_weights_4
        ),
        neural_biases_4
    )
    
    return neural_outputs_4
    



