import tensorflow as tf

a = tf.constant([[1, 2], [1, 2]])
b = tf.constant([[3], [4]])
c = tf.matmul(a, b)
d = tf.multiply(a, b)
with tf.Session() as sess:
    print(sess.run(c))
    print(sess.run(d))

