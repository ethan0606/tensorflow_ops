import tensorflow as tf

a = tf.constant([[1], [1]])
b = tf.constant([[2], [2]])
c = tf.concat([a, b], axis=1)

with tf.Session() as sess:
    print(sess.run(c))
