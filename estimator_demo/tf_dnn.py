import tensorflow as tf

from util import DataSetIterator

train_filenames = ['/Users/yifanguo/Desktop/tensorflow_ops/data/agaricus/tf_train']
test_filenames = ['/Users/yifanguo/Desktop/tensorflow_ops/data/agaricus/tf_test']

train_iter = DataSetIterator.get_iter(train_filenames, batch=40)
test_iter = DataSetIterator.get_iter(test_filenames, batch=10)

labels = tf.placeholder(shape=[None, 1], dtype=tf.float32)
features = tf.placeholder(shape=[None, 126], dtype=tf.float32)


def neural_net(inputs, initializer=tf.initializers.glorot_normal, activation=tf.nn.relu):
    h0 = tf.layers.dense(inputs, 50, kernel_initializer=initializer, activation=activation)
    h1 = tf.layers.dense(h0, 20, kernel_initializer=initializer, activation=activation)
    out = tf.layers.dense(h1, 1)
    return out


logits = neural_net(features)
loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits), axis=0)
score = tf.sigmoid(logits)
train_op = tf.train.AdamOptimizer(0.001).minimize(loss_op)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    i = 0
    while True:
        try:
            train_batch = sess.run(train_iter)
            test_batch = sess.run(test_iter)
            sess.run(train_op, feed_dict={features: train_batch['features'],
                                          labels: train_batch['label']})
            test_loss = sess.run(loss_op, feed_dict={features: test_batch['features'],
                                                     labels: test_batch['label']})
            print('test batch:%d' % i, 'loss:%f' % test_loss)

            i = i + 1
        except:
            print('train all data')
            break
