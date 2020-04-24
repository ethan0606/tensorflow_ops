import tensorflow as tf

a = tf.constant([1., 2., 3.])
b = tf.sigmoid(a)
c = tf.nn.sigmoid(a)
with tf.Session() as sess:
    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(c))

