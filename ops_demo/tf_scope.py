import tensorflow as tf

with tf.variable_scope('v1', reuse=tf.AUTO_REUSE):
    a = tf.get_variable(name='a', shape=(), dtype=tf.float32, initializer=tf.initializers.constant)
    b = tf.get_variable(name='a', shape=(), dtype=tf.float32)

print(a)
print(b)

with tf.name_scope('v2'):
    c = tf.get_variable(name='a', shape=(), dtype=tf.float32, initializer=tf.initializers.constant)
    d = tf.get_variable(name='b', shape=(), dtype=tf.float32)
    e = tf.Variable(name='c', shape=(), dtype=tf.float32, initial_value=1.)

print(c)
print(d)
print(e)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(c))
    print(sess.run(d))
