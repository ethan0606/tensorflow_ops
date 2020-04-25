import tensorflow as tf

a = tf.random.normal(shape=[3,5])
samples = tf.random.categorical(tf.math.log([[0.5, 0.5]]), 5)

with tf.Session() as sess:
    print(sess.run(a))
    print(sess.run(samples))
