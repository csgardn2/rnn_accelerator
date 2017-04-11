import tensorflow

# Inputs:
#   int neural_inputs[9]
#       A single 3 x 3 greyscale tile from a larger input image
#       flattened in row-major order
# Return Value:
#   Handle to the output layer of a tensorflow neural graph.  The output layer
#   is a prediction of the output pixel value
def gradient_basic_graph(neural_inputs):
    
    # Be advised, variables declared inside a python function are local, and
    # will be destroyed when the function returns.  However, it is OK for
    # these intermediate variables like neural_outputs_1 to be destroyed since
    # each command to tensorflow.blah.blah.blah constructs objectes which
    # are kept in tensorflow's runtime environment.  The python variable is just
    # a handle for us to use as a reference.
    
    # Layer 1
    # Output
    # Input dimensions = [9]
    # Output dimensions = [1]
    
    neural_weights = tensorflow.Variable(tensorflow.truncated_normal([9, 1], stddev = 0.1))
    neural_biases = tensorflow.Variable(tensorflow.truncated_normal([1], stddev = 0.1))
    return tensorflow.matmul(neural_inputs, neural_weights) + neural_biases
    

