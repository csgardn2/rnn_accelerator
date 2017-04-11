import sys
import tensorflow

# Note:
#   Variables declared inside a function are local, and are destroyed when
#   the function returns.  However, the item returned by tensorflow.Variable
#   and tensorflow.placeholder are meerly handles to objects declared inside
#   of tensorflow's runtime environment.  The neural topology persists outside
#   of this function even when the handle goes out of scope and is destroyed.
#
# Best accuracy = 80.81%
def histogram_softsign_graph(neural_inputs, inputs_per_sample, num_histogram_bins):
    
    # Tensorflow conventions
    # [height, width]
    
    # Layer 1
    # Input dimensions = [None, inputs_per_sample]
    # Output dimesnions = [None, 512]
    neural_weights_1 = tensorflow.Variable(tensorflow.zeros([inputs_per_sample, 512]))
    neural_biases_1 = tensorflow.Variable(tensorflow.zeros([1, 512]))
    neural_outputs_1 = tensorflow.add(
        tensorflow.matmul(
            neural_inputs,
            neural_weights_1
        ),
        neural_biases_1
    )
    
    # Layer 2
    # Input dimensions = [None, 512]
    # Output dimesnions = [None, 512]
    neural_outputs_2 = tensorflow.nn.softsign(neural_outputs_1)
    
    # Layer 3
    # Input dimensions = [None, 512]
    # Output dimesnions = [None, num_histogram_bins]
    neural_weights_3 = tensorflow.Variable(tensorflow.zeros([512, num_histogram_bins]))
    neural_biases_3 = tensorflow.Variable(tensorflow.zeros([1, num_histogram_bins]))
    neural_outputs_3 = tensorflow.add(
        tensorflow.matmul(
            neural_outputs_2,
            neural_weights_3
        ),
        neural_biases_3
    )
    
    return neural_outputs_3
    

