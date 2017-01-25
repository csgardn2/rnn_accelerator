import tensorflow as tf

# setup the parameters
# TODO: these can be modified
BATCH_SIZE = 100
LEARNING_RATE = 0.01
TRAINING_FRACTION = 0.8

# import input data, output golden data


# inputs
# each input has size 9
x = tf.placeholder(tf.float32, [None, 9])

# weights and biases
# TODO: initial values can be modified
W = tf.Variable(tf.zeros([9, 1]))
b = tf.Variable(tf.zeros(1))

# trained outputs
y = tf.add(tf.matmul(x, W), b) #TODO: is tf.add necessary?

# golden outputs
y_ = tf.placeholder(tf.float32, [None, 1])

# cost
cost = #TODO

# train
# TODO: train steps can be modified
# eg. momentum
trainstep = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

# init
init = tf.initialize_all_variables()

# session
sess = tf.session()
sess.run(init)

for i in xrange(): #TODO: how many iterations?
    sess.run(trainstep,
             feed_dict={x: , y_:}) #TODO: import the data

# evaluation

