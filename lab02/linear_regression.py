import tensorflow as tf

x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

#y = tf.add(tf.mul(x, W), b)
y = W * x_data + b
cost = tf.reduce_mean(tf.square(y - y_data))

learning_rate = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train)
    if (step % 20 == 0):
        print(step, sess.run(cost), sess.run(W), sess.run(b))
