import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]
'''
x_data = [[1, 1, 1, 1, 1], [0, 2, 0, 4, 0], [1, 0, 3, 0, 5]]
y_data = [1, 2, 3, 4, 5]
'''

x = tf.placeholder(tf.float32, [3, None])
y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([1, 3], -1.0, 1.0))
y_ = tf.matmul(W, x)

cost = tf.reduce_mean(tf.pow(y_ - y, 2))
alpha = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(alpha)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for step in range(2001):
    cost_val, _ = sess.run([cost, train], feed_dict={x:x_data, y:y_data})
    if (step % 20 == 0):
        print(step, cost_val, sess.run(W))

