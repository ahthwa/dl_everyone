import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data = xy[0:3]
y_data = xy[3:]

print(x_data)
print(y_data)

W = tf.Variable(tf.random_uniform([3, 3], -1.0, 1.0))

x = tf.placeholder(tf.float32, [3, None])
y = tf.placeholder(tf.float32, [3, None])

y_ = tf.nn.softmax(tf.matmul(W, x))
cost = - tf.reduce_mean(tf.reduce_sum(y * tf.log(y_)))
alpha = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(alpha)
train = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for step in range(2001):
        cost_val, w_val, _ = sess.run([cost, W, train], feed_dict = {x:x_data, y:y_data})
        if (step % 20 == 0):
            print(step, cost_val, w_val)

'''
# prediction
print(sess.run(y_, feed_dict = {X:[[1, 1, 1, 1], [5, 2, 4, 3], [3, 2, 3, 5]]}) > 0.5)
'''

sess.close()
