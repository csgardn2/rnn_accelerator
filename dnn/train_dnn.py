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
import dataset
import hotspot.hotspot as bm # TODO

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.0001, 'learning rate')
flags.DEFINE_integer('batch_size', 100, 'batch size')
flags.DEFINE_integer('max_steps', 9100, 'max training steps')
flags.DEFINE_integer('hidden1',
        0, 'number of neurons in hidden layer 1')
flags.DEFINE_integer('hidden2',
        0, 'number of layers in hidden layer 2')
flags.DEFINE_string('data_dir',
        '/home/cosine/research/rnn_accelerator/dnn/hotspot/data/', 'data directory')
flags.DEFINE_string('input_data_type',
        'float', 'input data type: float or int')
flags.DEFINE_string('output_data_type',
        'float', 'output data type: float or int')
flags.DEFINE_bool('separate_file',
        False, 'indicates whether training, validation, testing data are in separate files')
flags.DEFINE_string('log_dir',
        '/home/cosine/research/rnn_accelerator/dnn/hotspot/log', 'directory to put the log data')
#for hotspot training
flags.DEFINE_string('tile_size',
        1, 'tile size')
flags.DEFINE_string('num_maps',
        2, 'number of maps used')


def run_training():
    '''train the Neural Network'''
    # sanity check
    assert(FLAGS.input_data_type == 'float'
            or FLAGS.input_data_type == 'int')
    assert(FLAGS.output_data_type == 'float'
            or FLAGS.output_data_type == 'int')
    # import the dataset
    '''
    data_sets = dataset.Datasets(FLAGS.data_dir,
            FLAGS.separate_file,
            FLAGS.input_data_type, FLAGS.output_data_type)
    '''
    #for hotspot training
    data_sets = dataset.Datasets(FLAGS.data_dir,
            FLAGS.separate_file,
            FLAGS.input_data_type, FLAGS.output_data_type,
            FLAGS.tile_size, FLAGS.num_maps)

    with tf.Graph().as_default():
        # placeholder
        input_pl, golden_pl = util.generate_placeholder(
                data_sets.num_in_neuron,
                data_sets.num_out_neuron,
                FLAGS.batch_size,
                FLAGS.input_data_type,
                FLAGS.output_data_type
                )
        # build graph
        if FLAGS.hidden1 == 0:
            assert(FLAGS.hidden2 == 0)
            outputs = util.layer('output_layer', input_pl,
                    data_sets.num_in_neuron, data_sets.num_out_neuron, None)
        else:
            hidden1 = util.layer('hidden1', input_pl,
                    data_sets.num_in_neuron, FLAGS.hidden1, tf.nn.relu)
            if FLAGS.hidden2 == 0:
                outputs = util.layer('output_layer', hidden1,
                        FLAGS.hidden1, data_sets.num_out_neuron, None)
            else:
                hidden2 = util.layer('hidden2', hidden1,
                        FLAGS.hidden1, FLAGS.hidden2, tf.nn.relu)
                outputs = util.layer('output_layer', hidden2,
                        FLAGS.hidden2, data_sets.num_out_neuron, None)

        # loss
        loss = bm.loss(outputs, golden_pl)

        # train
        train_op = bm.training(loss, FLAGS.learning_rate)

        # accumulated error for one batch of data
        error = bm.error(outputs, golden_pl)

        # summary - not necessary
        summary = tf.merge_all_summaries()

        # init
        init = tf.initialize_all_variables()

        # sess
        sess = tf.Session()

        # summary writer - not necessary
        summary_writer = tf.train.SummaryWriter(FLAGS.log_dir, sess.graph)

        # everything built, run init
        sess.run(init)

        # start training
        #_, max_steps = data_sets.train.max_steps(FLAGS.batch_size)
        for step in xrange(FLAGS.max_steps):
            feed_dict = util.fill_feed_dict(data_sets.train,
                    input_pl, golden_pl,
                    FLAGS.batch_size)
            sess.run(train_op, feed_dict=feed_dict)

            # print the loss every 100 steps
            # write the summary
            # evaluate the model
            if not step % 100:
                print('step %d: loss = %.2f' % (step,
                    sess.run(loss, feed_dict=feed_dict) ))

                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
                '''
                print('training data evaluation')
                util.do_eval(sess, error,
                        input_pl, golden_pl,
                        FLAGS.batch_size, data_sets.train)
                '''
                print('validation data evaluation')
                util.do_eval(sess, error,
                        input_pl, golden_pl,
                        FLAGS.batch_size, data_sets.validate)

        # final accuracy
        print('test data evaluation')
        util.do_eval(sess, error,
        input_pl, golden_pl,
        FLAGS.batch_size, data_sets.test)

def main(_):
    run_training()

if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--learning_rate',
            type=float,
            default=0.01,
            help='learning rate'
    )
    parser.add_argument(
            '--max_steps',
            type=int,
            default=2000,
            help='max training step'
    )
    parser.add_argument(
            '--batch_size',
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
            '--data_dir',
            type=str,
            default='/home/cosine/research/rnn_accelerator/dnn/hotspot/data/',
            help='directory of data'
    )
    parser.add_argument(
            '--input_data_type',
            type=str,
            default='float',
            help='type of input data, choose from "int", "float", "double"'
    )
    parser.add_argument(
            '--output_data_type',
            type=str,
            default='float',
            help='type of output data, choose from "int", "float", "double"'
    )
    parser.add_argument(
            '--separate_file',
            type=bool,
            default=False,
            help='indicates whether training, validation, testing data are in separate files'
    )
    parser.add_argument(
            '--log_dir',
            type=str,
            default='/home/cosine/research/rnn_accelerator/dnn/hotspot/log',
            help='Directory to put the log data'
    )
    '''
    tf.app.run()
