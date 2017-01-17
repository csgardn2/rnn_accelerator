import tensorflow

# Inputs:
#   int neural_inputs[784]
#       A single 28 x 28 greyscale image of a handwritten digit [0 - 9]
#       flattened in row-major order
# Return Value:
#   Handle to the output layer of a tensorflow neural graph.  The output layer
#   is a list of 10 floats where each float is the probability that the input
#   digit is the corresponding index.
def mnist_basic_graph(neural_inputs):
    
    # Be advised, variables declared inside a python function are local, and
    # will be destroyed when the function returns.  However, it is OK for
    # these intermediate variables like neural_outputs_1 to be destroyed since
    # each command to tensorflow.blah.blah.blah constructs objectes which
    # are kept in tensorflow's runtime environment.  The python variable is just
    # a handle for us to use as a reference.
    
    # Layer 1
    # Output
    # Input dimensions = [784]
    # Output dimensions = [10]
    neural_weights = tensorflow.Variable(tensorflow.truncated_normal([784, 10], stddev = 0.1))
    neural_biases = tensorflow.Variable(tensorflow.truncated_normal([10], stddev = 0.1))
    return tensorflow.matmul(neural_inputs, neural_weights) + neural_biases
    

