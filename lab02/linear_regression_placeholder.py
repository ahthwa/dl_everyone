import tensorflow as tf

x_data = [1., 2., 3.]
y_data = [1., 2., 3.]

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

y_ = W * x + b

cost = tf.reduce_mean(tf.square(y_ - y))
learning_rate = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for step in range(2001):
    #sess.run(train, feed_dict={x:x_data, y:y_data})
    cost_val, _ = sess.run([cost, train], feed_dict={x:x_data, y:y_data})

    if (step % 20 == 0):
        print(step, cost_val, sess.run(W), sess.run(b))

# prediction
print(sess.run(y_, feed_dict={x:5}))
sess.close()

