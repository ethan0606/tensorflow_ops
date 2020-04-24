import tensorflow as tf

a = tf.get_variable(name='a', shape=(), dtype=tf.float32, initializer=tf.initializers.glorot_normal)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(a))
