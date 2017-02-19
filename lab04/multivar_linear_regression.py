import tensorflow as tf

x1_data = [1, 0, 3, 0, 5]
x2_data = [0, 2, 0, 4, 0]
y_data = [1, 2, 3, 4, 5]

w1 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
w2 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

y_ = tf.multiply(w1, x1) + tf.multiply(w2, x2) + b

cost = tf.reduce_mean(tf.pow(y_ - y, 2))
alpha = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(alpha)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for step in range(2001):
    cost_val, _ = sess.run([cost, train], feed_dict={x1:x1_data, x2:x2_data, y:y_data})
    if (step % 20 == 0):
        print(step, cost_val, sess.run(w1), sess.run(w2), sess.run(b))

