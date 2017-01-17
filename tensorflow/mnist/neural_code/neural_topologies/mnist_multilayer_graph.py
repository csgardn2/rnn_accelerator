import tensorflow

# Inputs:
#   int neural_inputs[784]
#       A single 28 x 28 greyscale image of a handwritten digit [0 - 9]
#       flattened in row-major order
# Return Value:
#   Handle to the output layer of a tensorflow neural graph.  The output layer
#   is a list of 10 floats where each float is the probability that the input
#   digit is the corresponding index.
def mnist_multilayer_graph(neural_inputs):
    
    # Be advised, variables declared inside a python function are local, and
    # will be destroyed when the function returns.  However, it is OK for
    # these intermediate variables like neural_outputs_1 to be destroyed since
    # each command to tensorflow.blah.blah.blah constructs objectes which
    # are kept in tensorflow's runtime environment.  The python variable is just
    # a handle for us to use as a reference.
    
    # Layer 1
    # Convolution
    # Input dimensions = [784]
    # Output dimensions = [28, 28, 1, 32]
    reshaped_inputs = tensorflow.reshape(neural_inputs, [-1, 28, 28, 1])
    neural_weights_1 = tensorflow.Variable(tensorflow.truncated_normal([5, 5, 1, 32], stddev = 0.1));
    neural_biases_1 = tensorflow.Variable(tensorflow.truncated_normal([32], stddev = 0.1))
    neural_outputs_1 = tensorflow.nn.conv2d(
        reshaped_inputs,
        neural_weights_1,
        strides = [1, 1, 1, 1],
        padding = 'SAME'
    ) + neural_biases_1
    
    # Layer 2
    # Max pooling
    # Input dimensions = [28, 28, 1, 32]
    # Output dimensions = [14, 14, 1, 32]
    neural_outputs_2 = tensorflow.nn.max_pool(
        neural_outputs_1,
        ksize = [1, 2, 2, 1],
        strides = [1, 2, 2, 1],
        padding = 'SAME'
    )
    
    # Layer 3
    # Convolution
    # Input dimensions = [14, 14, 1, 32]
    # Output dimensions = [14, 14, 1, 64]
    neural_weights_3 = tensorflow.Variable(tensorflow.truncated_normal([5, 5, 32, 64], stddev = 0.1))
    neural_biases_3 = tensorflow.Variable(tensorflow.truncated_normal([64], stddev = 0.1))
    neural_outputs_3 = tensorflow.nn.conv2d(
        neural_outputs_2,
        neural_weights_3,
        strides = [1, 1, 1, 1],
        padding = 'SAME'
    ) + neural_biases_3
    
    # Layer 4
    # Max pooling
    # Input dimensions = [14, 14, 1, 64]
    # Output dimensions = [7, 7, 1, 64]
    neural_outputs_4 = tensorflow.nn.max_pool(
        neural_outputs_3,
        ksize = [1, 2, 2, 1],
        strides = [1, 2, 2, 1],
        padding = 'SAME'
    )
    
    # Layer 5
    # Fully connected
    # Input dimensions = [7, 7, 1, 64]
    # Output dimensions = [1024]
    reshaped_outputs_4 = tensorflow.reshape(neural_outputs_4, [-1, 7 * 7 * 64])
    neural_weights_5 = tensorflow.Variable(tensorflow.truncated_normal([7 * 7 * 64, 1024], stddev = 0.1))
    neural_biases_5 = tensorflow.Variable(tensorflow.truncated_normal([1024], stddev = 0.1))
    neural_outputs_5 = tensorflow.matmul(reshaped_outputs_4, neural_weights_5) + neural_biases_5
    
    # Layer 6
    # Dropout
    # Input dimensions = [1024]
    # Output dimensinos = [1024]
#    neural_outputs_6 = tensorflow.nn.dropout(neural_outputs_5, tensorflow.placeholder(tensorflow.float32))
    neural_outputs_6 = neural_outputs_5
    
    # Layer 7
    # Output
    # Input dimensions = [1024]
    # Output dimensions = [1024]
    neural_weights_7 = tensorflow.Variable(tensorflow.truncated_normal([1024, 10], stddev = 0.1))
    neural_biases_7 = tensorflow.Variable(tensorflow.truncated_normal([10], stddev = 0.1))
    return tensorflow.matmul(neural_outputs_6, neural_weights_7) + neural_biases_7

