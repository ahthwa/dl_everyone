import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)
c = a + b

sess = tf.Session()
print(c)
print(sess.run(c))
sess.close()
