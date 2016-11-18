import tensorflow as tf
import input_data
import random

learning_rate = 0.01
training_epochs = 5
batch_size = 100
display_step = 1

mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.random_uniform([784, 10], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1, 10], -1.0, 1.0))

y_ = tf.nn.softmax(tf.matmul(x, W) + b)
cost = tf.reduce_mean( - tf.reduce_sum( y * tf.log(y_), reduction_indices=1))

alpha = tf.Variable(.1)
optimizer = tf.train.GradientDescentOptimizer(alpha)
train = optimizer.minimize(cost)

total_batch = int(mnist.train.num_examples / batch_size)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
#saver = tf.train.Saver()

for epoch in range(training_epochs):
    avg_cost = 0.
    for i in range(total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        cost_val, w_val, _ = sess.run([cost, W, train], feed_dict = {x:batch_x, y:batch_y})
        avg_cost += cost_val / total_batch

    if (epoch % display_step == 0):
        print(epoch, avg_cost)

r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
print("Prediction: ", sess.run(tf.argmax(y_, 1), {x:mnist.test.images[r:r+1]}))

correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#print("Accuracy: ", accuracy.eval({x:mnist.test.images, y:mnist.test.labels})) # default session이 필요하다.
print("Accuracy: ", sess.run(accuracy, {x:mnist.test.images, y:mnist.test.labels}))
sess.close()
