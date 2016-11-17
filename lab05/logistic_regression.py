import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

X = tf.placeholder(tf.float32, [3, None])
Y = tf.placeholder(tf.float32)
W = tf.Variable(tf.random_uniform([1, 3], -1.0, 1.0))

#y_ = 1 / (1 + tf.exp(-tf.matmul(W, X)))
y_ = tf.div(1., 1. + tf.exp(-tf.matmul(W, X)))
cost = - tf.reduce_mean( Y * tf.log(y_) + (1 - Y)*tf.log(1-y_))

alpha = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(alpha)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    cost_val, w_val, _ = sess.run([cost, W, train], feed_dict = {X:x_data, Y:y_data})
    if (step % 20 == 0):
        print(step, cost_val, w_val)

# prediction
print(sess.run(y_, feed_dict = {X:[[1, 1, 1, 1], [5, 2, 4, 3], [3, 2, 3, 5]]}) > 0.5)

sess.close()
