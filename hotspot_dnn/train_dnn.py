'''
Arthor:   Yu-Hsuan Tseng
Date:     02/02/2017
Description: train with arbitrary hidden layers
'''
import argparse
import sys
import tensorflow as tf
import util
# import benchmark and corresponding dataset
from hotspot_dnn import dataset
import hotspot_dnn.hotspot as bm

FLAGS = None

def run_training():
    '''train the Neural Network'''
    # import the dataset
    data_sets, num_in, num_gold, type_in, type_gold = dataset.read_data(FLAGS.input_data_dir)

    with tf.Graph().as_default():
        # placeholder
        input_pl, golden_pl = util.generate_placeholder(
                num_in, num_gold, FLAGS.batch_size,
                type_in, type_gold
                )
        # build graph
        if FLAGS.hidden1 == 0:
            assert(FLAGS.hidden2 == 0)
            outputs = util.layer('output layer', input_pl, num_gold, None)
        else:
            hidden1 = util.layer('hidden1', input_pl, FLAGS.hidden1, tf.nn.relu)
            if FLAGS.hidden2 == 0:
                outputs = util.layer('output layer', hidden1, num_gold, None)
            else:
                hidden2 = util.layer('hidden2', hidden1, FLAGS.hidden2, tf.nn.relu)
                outputs = util.layer('output layer', hidden2, num_gold, None)
        # loss

def main(_):
    run_training()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--learning_rate',
            type=float,
            default=0.01,
            help='learning rate'
    )
    parser.add_argument(
            'batch_size',
            type=int,
            default=100,
            help='batch size'
    )
    parser.add_argument(
            '--hidden1',
            type=int,
            default=32,
            help='number of neurons in hidden layer one'
    )
    parser.add_argument(
            '--hidden2',
            type=int,
            default=0,
            help='number of neurons in hidden layer two'
    )
    parser.add_argument(
            '--input_data_dir',
            type=str,
            default='/home/cosine/rnn_accelerator/heat_maps/simulation_output_temperatures/256x256_random_iterations0000to0031/source_256x256_random_iteration0001.csv',
            help='directory of input data'
    )
    parser.add_argument(
            '--label_data_dir',
            type=str,
            default='/home/cosine/rnn_accelerator/heat_maps/simulation_output_temperatures/256x256_random_iterations0000to0031/destination_256x256_random_iteration0001.csv',
            help='directory of label (golden output) data'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
