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

FLAGS = None

def run_training():
    '''train the Neural Network'''
    # import the dataset
    data_sets = dataset.datasets(FLAGS.data_dir, FLAGS.separate_file,
            FLAGS.input_data_type, FLAGS.output_data_type)

    with tf.Graph().as_default():
        # placeholder TODO
        input_pl, golden_pl = util.generate_placeholder(
                len(data_sets.train.input_data),
                FLAGS.batch_size,
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
        loss = bm.loss(outputs, golden_pl)

        # train
        train_op = bm.training(loss, FLAGS.learning_rate)

        # accumulated error for one batch of data
        error = bm.error(outputs, golden_pl)

        # summary - not necessary
        summary = tf.summary.merge_all()

        # init
        init = tf.initialize_all_variables()

        # sess
        sess = tf.Session()

        # summary writer - not necessary
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        # everything built, run init
        sess.run(init)

        # start training
        _, max_steps = data_sets.train.max_steps(batch_size)
        for step in xrange(max_steps):
            feed_dict = util.fill_feed_dict(data_sets.train,
                    input_pl, golden_pl,
                    FLAGS.batch_size)
            sess.run(train_op, feed_dict=feed_dict)

            # print the loss every 100 steps
            # write the summary
            # evaluate the model
            if not step % 100:
                print('step %d: loss = %.2f' % (step, loss))

                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

                print('training data evaluation')
                util.do_eval(sess, error,
                        input_pl, golden_pl,
                        FLAGS.batch_size, data_sets.train)
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--learning_rate',
            type=float,
            default=0.01,
            help='learning rate'
    )
    '''
    parser.add_argument(
            '--max_steps',
            type=int,
            default=2000,
            help='max training step'
    )
    '''
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
            '--data_dir',
            type=str,
            default='/home/cosine/research/rnn_accelerator/heat_maps/simulation_output_temperatures/256x256_random_iterations0000to0031/',
            help='directory of data'
    )
    parser.add_argument(
            '--input_data_type',
            type=str,
            default='float'
            help='type of input data, choose from "int", "float", "double"'
    )
    parser.add_argument(
            '--output_data_type',
            type=str,
            default='float'
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
            typr=str,
            default='/home/cosine/research/rnn_accelerator/dnn/hotspot/log'
            help='Directory to put the log data'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
