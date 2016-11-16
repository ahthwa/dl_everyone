import tensorflow as tf

x_data = [1., 2., 3.]
y_data = [1., 2., 3.]

W = tf.placeholder(tf.float32)

#y= W * x_data
y= tf.mul(W, x_data)
cost = tf.reduce_sum(tf.pow(y_data - y, 2)) / len(x_data)
init = tf.initialize_all_variables()

sess = tf.Session()

sess.run(init)

w_val = []
cost_val = []

for i in range(-30, 50):
    c_val = sess.run(cost, feed_dict={W:i*0.1})
    w_val.append(i * 0.1)
    cost_val.append(c_val)
    print(i * 0.1, c_val)

sess.close()

'''
plt.plot(w_val, cost_val, 'ro')
plt.ylabel('Cost')
plt.xlabel('W')
plt.show()
'''
