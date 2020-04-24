import tensorflow as tf

print(tf.VERSION)

a = tf.placeholder(tf.float32, shape=[1, 2], name='a')

with tf.Session() as sess:
    print(sess.run(a, feed_dict={a: [[1., 2.]]}))
