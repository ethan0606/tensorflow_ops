import tensorflow as tf

a = tf.constant([[1, 2.2, 3], [3, 4, 5]])
b = tf.reduce_mean(a, axis=0)
c = tf.reduce_mean(a, axis=1)

with tf.Session() as sess:
    print(sess.run(b))
    print(sess.run(c))

