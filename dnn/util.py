'''
Arthor:      Yu-Hsuan Tseng
Date:        02/02/2017
Description: this file contains utility functions used by any
feed-forward neural network
'''
import tensorflow as tf

def generate_placeholder(num_in, num_out, batch_size, type_in, type_out):
    '''generate placeholder for inputs and golden output

    Args:
        num_in: number of input neurons
        num_out: number of output neurons
        batch_size: batch size
        type_in: type of inputs, e.g. float
        type_out: type of outputs

    Returns:
        input_pl: input placeholder
        golden_pl: output placeholder
    '''
    # type
    assert(type_in == "int" or type_in == "float")
    assert(type_out == "int" or type_out == "float")
    type_in_tf = tf.float32
    if type_in == "int":
        type_in_tf = tf.int32
    type_out_tf = tf.float32
    if type_out == "int":
        type_out_tf = tf.int32
    # placeholder
    input_pl = tf.placeholder(type_in_tf,
            shape=(batch_size, num_in))
    golden_pl = tf.placeholder(type_out_tf,
            shape=(batch_size, num_out))
    return input_pl, golden_pl

def fill_feed_dict(data_set, input_pl, golden_pl, batch_size):
    '''fill the feed_dict

    Args:
        data_set: input and output dataset
        input_pl: input placeholder
        golden_pl: golden output placeholder
        batch_size: batch size

    Returns:
        feed_dict: the feed dictionary mapping from placeholders to values
    '''
    input_feed, golden_feed = data_set.next_batch(batch_size)
    feed_dict = {
        input_pl: input_feed,
        golden_pl: golden_feed
    }
    # debug
    #print input_pl.get_shape()
    #print len(input_feed)
    return feed_dict

def layer(name, input_units, num_in, num_out, activation_function):
    '''calculation within a layer

    Args:
        name: the name of this layer
        input_units: input placeholder (type: Tensor)
        num_in: number of input neurons(neurons in the previous layer)
        num_out: number of output neurons(neurons within this layer)
        activation_function: the activation_function applied on outputs,
        None if nothing needs to be done

    Returns:
        output_units: output neurons(neurons within this layer)
    '''
    with tf.name_scope(name):
        # TODO: weights and biases initialization can be changed
        weights = tf.Variable(tf.zeros([num_in, num_out]),
                name='weights')
        biases = tf.Variable(tf.zeros([num_out]),
                name='biases')
        output_units = tf.matmul(input_units, weights) + biases
        if activation_function:
            output_units = activation_function(output_units)
    return output_units

def do_eval(sess, error,
        input_pl, golden_pl,
        batch_size, data_set):
    '''evaluate and print the accuracy for the given whole dataset

    Args:
        sess: the session in which the model has been trained
        error: the error for one batch of data (from benchmark)
        input_pl: input placeholder
        golden_pl: golden output placeholder
        batch_size: batch size
        data_set: the data to be evaluated
    '''
    error_sum = 0
    data_set.reset_touched()
    # steps_per_epoch is floor(data_size/batch_size)
    # num_examples is steps_per_epoch * batch_size
    num_examples, steps_per_epoch = data_set.max_steps(batch_size)
    for x in xrange(steps_per_epoch):
        #debug
        #print "iteration: %d" %x
        feed_dict = fill_feed_dict(data_set,
                input_pl, golden_pl,
                batch_size)
        error_sess = sess.run(error, feed_dict=feed_dict)
        #debug
        #print "error = %f" %error_sess
        error_sum = error_sum + error_sess
    accuracy = 100 - 100 * float(error_sum) / float(num_examples)
    print('Number of examples: %d, Accuracy: %.3f'
            % (num_examples, accuracy))
