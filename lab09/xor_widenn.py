import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt', unpack=True)
x_data = xy[0:-1]
y_data = xy[-1]

X = tf.placeholder(tf.float32, [3, None])
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([10, 3], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([1, 10], -1.0, 1.0))
b2 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

l1 = tf.div(1., 1. + tf.exp( - (tf.matmul(W1, X))))
#y_ = tf.sigmoid(tf.matmul(W2, l1) + b2)
y_ = tf.div(1., 1. + tf.exp( - (tf.matmul(W2, l1) + b2)))

cost = - tf.reduce_mean(Y * tf.log(y_) + (1 - Y) * tf.log(1-y_))
alpha = 0.1
optimizer = tf.train.GradientDescentOptimizer(alpha)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    for step in range(10001):
        cost_val, weight1_val, weight2_val, _ = sess.run([cost, W1, W2, train], feed_dict = {X:x_data, Y:y_data})
        if (step % 100 == 0):
            print(step, cost_val, weight1_val, weight2_val)

    correct_prediction = tf.equal(tf.floor(y_ + 0.5), Y)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(sess.run([y_, tf.floor(y_ + 0.5), correct_prediction, accuracy], feed_dict={X:x_data, Y:y_data}))
    print("Accuracy:", accuracy.eval({X:x_data, Y:y_data}))
