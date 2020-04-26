import tensorflow as tf

a = tf.constant(1., dtype=tf.float32)
b = tf.cast(a, dtype=tf.int32)

with tf.Session() as sess:
    print(sess.run(a))
    print(sess.run(b))
