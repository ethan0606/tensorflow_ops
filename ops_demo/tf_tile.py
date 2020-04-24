import tensorflow as tf

a = tf.constant([[1, 2], [3, 4]])
b = tf.tile(a, [3, 2])

with tf.Session() as sess:
    print(sess.run(a))
    print(sess.run(b))
