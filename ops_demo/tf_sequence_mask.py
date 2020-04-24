import tensorflow as tf

a = tf.sequence_mask([1, 2, 3], 5)
b = tf.sequence_mask([[1, 2], [3, 4]])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(a))
    print(sess.run(b))
