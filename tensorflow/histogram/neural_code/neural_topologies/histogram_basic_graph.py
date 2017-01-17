import sys
import tensorflow

# Note:
#   Variables declared inside a function are local, and are destroyed when
#   the function returns.  However, the item returned by tensorflow.Variable
#   and tensorflow.placeholder are meerly handles to objects declared inside
#   of tensorflow's runtime environment.  The neural topology persists outside
#   of this function even when the handle goes out of scope and is destroyed.
def histogram_basic_graph(neural_inputs, inputs_per_sample, num_histogram_bins):
    
    # Tensorflow conventions
    # [height, width]
    
    # Layer 1
    # Input dimensions = [NONE, inputs_per_sample]
    # Output dimesnions = [NONE, num_histogram_bins]
    neural_weights = tensorflow.Variable(tensorflow.zeros([inputs_per_sample, num_histogram_bins]))
    neural_biases = tensorflow.Variable(tensorflow.zeros([1, num_histogram_bins]))
    
    neural_outputs = tensorflow.add(
        tensorflow.matmul(
            neural_inputs,
            neural_weights
        ),
        neural_biases
    )
    
    return neural_outputs
    

