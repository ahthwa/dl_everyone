import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data = xy[0:3].transpose()
y_data = xy[3:].transpose()

print(x_data)
print(y_data)

W = tf.Variable(tf.random_uniform([3, 3], -1.0, 1.0))
x = tf.placeholder(tf.float32, [None, 3])
y = tf.placeholder(tf.float32, [None, 3])

y_ = tf.nn.softmax(tf.matmul(x, W))
cost = tf.reduce_mean(- tf.reduce_sum( y * tf.log(y_), 1))

alpha = tf.Variable(0.01)
optimizer = tf.train.GradientDescentOptimizer(alpha)
train = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for step in range(2001):
        cost_val, w_val, _ = sess.run([cost, W, train], feed_dict = {x:x_data, y:y_data})
        if (step % 200 == 0):
            print(step, cost_val, w_val)
    # prediction
    print(sess.run([tf.arg_max(y_, 1), y_], feed_dict = {x:[[1, 11, 7], [1, 3, 4], [1, 1, 0], [1, 3, 5]]}))


