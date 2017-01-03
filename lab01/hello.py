import tensorflow as tf

hello = tf.constant('hello tensorflow')

print(hello)

sess = tf.Session()
print(sess.run(hello))
sess.close()
