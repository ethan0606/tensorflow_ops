import tensorflow as tf

voc_size = 10
emb_size = 2

a = tf.get_variable(shape=[voc_size, emb_size], name='a', dtype=tf.float32, initializer=tf.initializers.glorot_normal)
p = tf.placeholder(shape=(), dtype=tf.int32)
b = tf.nn.embedding_lookup(a, p)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(a))
    a = sess.run(b, feed_dict={p: 1})
    b = sess.run(b, feed_dict={p: 9})
    print(a)
    print(b)
