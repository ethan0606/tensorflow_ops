import tensorflow as tf

a = tf.constant([1, 2, 3, 4])
b = tf.reshape(shape=[2, 2], tensor=a)
with tf.Session() as sess:
    print(sess.run(a))
    print(sess.run(b))
