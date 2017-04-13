import sys
import tensorflow

# Note:
#   Variables declared inside a function are local, and are destroyed when
#   the function returns.  However, the item returned by tensorflow.Variable
#   and tensorflow.placeholder are meerly handles to objects declared inside
#   of tensorflow's runtime environment.  The neural topology persists outside
#   of this function even when the handle goes out of scope and is destroyed.
def fourier_basic_graph(neural_inputs, samples_per_function, coefficients_per_function):
    
    # Tensorflow conventions
    # [height, width]
    
    # Layer 1
    # Input dimensions = [NONE, samples_per_function]
    # Output dimesnions = [NONE, coefficients_per_function]
    neural_weights = tensorflow.Variable(tensorflow.zeros([samples_per_function, coefficients_per_function]))
    neural_biases = tensorflow.Variable(tensorflow.zeros([1, coefficients_per_function]))
    
    neural_outputs = tensorflow.add(
        tensorflow.matmul(
            neural_inputs,
            neural_weights
        ),
        neural_biases
    )
    
    return neural_outputs
    

