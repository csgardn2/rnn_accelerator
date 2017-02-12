'''
Arthor:      Yu-Hsuan Tseng
Date:        02/02/2017
Description: this file contains utility functions used by any
feed-forward neural network
'''
import tensorflow as tf

def generate_placeholder(num_in, num_gold, batch_size, type_in, type_gold):
    '''generate placeholder for inputs and golden output

    Args:
        num_in: number of inputs
        num_gold: number of golden outputs (same size as output)
        batch_size: batch size
        type_in: type of inputs, eg. tf.float32
        type_gold: type of outputs

    Returns:
        input_pl: input placeholder
        golden_pl: output placeholder
    '''

    input_pl = tf.placeholder(type_in, shape=(batch_size, num_in))
    golden_pl = tf.placeholder(type_out, shape=(batch_size, num_gold))
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
    return feed_dict

def layer(name, input_units, num_out, activation_function):
    '''calculation within a layer

    Args:
        name: the name of this layer
        input_units: input placeholder (type: list)
        num_out: number of output neurons(neurons within this layer)
        activation_function: the activation_function applied on outputs,
        None if nothing needs to be done

    Returns:
        output_units: output neurons(neurons within this layer)
    '''
    with tf.name_scope(name):
        num_in = len(input_units)
        # TODO: weights and biases initialization can be changed
        weights = tf.Variable(tf.zeros([num_in, num_out]),
                name='weights')
        biases = tf.Variable(tf.zeros([num_out]),
                name='biases')
        output_units = tf.matmul(input_units, weights) + biases
        if activation_function != 'None':
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
    # steps_per_epoch is floor(data_size/batch_size)
    # num_examples is steps_per_epoch * batch_size
    num_examples, steps_per_epoch = data_set.max_steps(batch_size)
    for x in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,
                input_pl, golden_pl,
                batch_size)
        error_sum = error_sum + sess.run(error, feed_dict=feed_dict)
    accuracy = 100 - 100 * float(error_sum) / float(num_examples)
    print('Number of examples: %d, Accuracy: %.3f',
            % (num_examples, accuracy))
