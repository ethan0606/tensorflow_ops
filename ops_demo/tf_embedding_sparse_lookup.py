import tensorflow as tf
import numpy as np

example = np.arange(24).reshape(6, 4).astype(np.float32)
print(example)
embedding = tf.Variable(example)

feature_size = 3
voc_dim = 5
idx = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 1], [1, 2], [2, 0]],
                      values=[0, 1, 2, 3, 0], dense_shape=[feature_size, voc_dim])

embed = tf.nn.embedding_lookup_sparse(embedding, idx, None, combiner='sqrtn')
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(embed))
print(sess.run(idx))
