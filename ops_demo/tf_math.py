import tensorflow as tf

a = tf.constant(1.)
print(a.shape)
b = tf.math.log1p(a)
with tf.Session() as sess:
    print(sess.run(a))
    print(sess.run(b))
