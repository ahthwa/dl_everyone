import tensorflow as tf

x_data = [1., 2., 3.]
y_data = [1., 2., 3.]

W = tf.Variable(tf.random_uniform([1], -10.0, 10.0))
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

y= W * X
cost = tf.reduce_sum(tf.pow(Y - y, 2)) / len(x_data)
init = tf.initialize_all_variables()

descent = W - tf.multiply(0.1, tf.reduce_mean(tf.multiply(tf.multiply(W, x_data) - Y, x_data)))
update = W.assign(descent)

sess = tf.Session()
sess.run(init)

for step in range(20):
    cost_val, w_val, _ = sess.run([cost, W, update], feed_dict = {X:x_data, Y:y_data})
    print(step, cost_val, w_val)
