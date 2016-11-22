import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt', unpack=True)
x_data = xy[0:-1]
y_data = xy[-1]

with tf.name_scope('model') as scope:
    X = tf.placeholder(tf.float32, [3, None])
    Y = tf.placeholder(tf.float32)

    W1 = tf.Variable(tf.random_uniform([2, 3], -1.0, 1.0), name = 'WEIGHT01')
    W2 = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
    b2 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

    l1 = tf.div(1., 1. + tf.exp( - (tf.matmul(W1, X))))
    y_ = tf.div(1., 1. + tf.exp( - (tf.matmul(W2, l1) + b2)))

tf.histogram_summary('weight 1', W1)
tf.histogram_summary('weight 2', W2)
tf.histogram_summary('bias 2', b2)
tf.histogram_summary('y_', y_)

# sigmoid 함수를 사용
#y_ = tf.sigmoid(tf.matmul(W2, l1) + b2)

# bias를 matrix안에
#W2 = tf.Variable(tf.random_uniform([1, 3], -1.0, 1.0))
#y_ = tf.div(1., 1. + tf.exp( - (tf.matmul(W2, l1_out))))
#l1_out = tf.concat(0, [l1, tf.ones([1, len(x_data[0])])])

with tf.name_scope('c') as scope:
    cost = - tf.reduce_mean(Y * tf.log(y_) + (1 - Y) * tf.log(1-y_))
    alpha = 0.1
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train = optimizer.minimize(cost)
    correct_prediction = tf.equal(tf.floor(y_ + 0.5), Y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
tf.scalar_summary('cost', cost)
tf.scalar_summary('accuracy', accuracy)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    merged_summary = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter('log', sess.graph)

    for step in range(10001):
        cost_val, weight1_val, weight2_val, _ = sess.run([cost, W1, W2, train], feed_dict = {X:x_data, Y:y_data})
        if (step % 1000 == 0):
            #print(step, cost_val, weight1_val, weight2_val)
            summary = sess.run(merged_summary, feed_dict = {X:x_data, Y:y_data})
            summary_writer.add_summary(summary, step)

