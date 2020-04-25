import tensorflow as tf
from util import DataSetIterator

train_filenames = ['/Users/yifanguo/Desktop/tensorflow_ops/data/agaricus/tf_train']
test_filenames = ['/Users/yifanguo/Desktop/tensorflow_ops/data/agaricus/tf_test']


def neural_net(inputs, initializer=tf.initializers.glorot_normal, activation=tf.nn.relu):
    h0 = tf.layers.dense(inputs, 50, kernel_initializer=initializer, activation=activation)
    h1 = tf.layers.dense(h0, 20, kernel_initializer=initializer, activation=activation)
    out = tf.layers.dense(h1, 1)
    return out


def model_fn(features, labels, mode):
    logits = neural_net(features['features'])
    score = tf.sigmoid(logits)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=score)
    loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits), axis=0)
    train_op = tf.train.AdamOptimizer(0.001).minimize(loss_op, global_step=tf.train.get_global_step())
    auc_op = tf.metrics.auc(labels=labels, predictions=score)

    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=score,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'auc': auc_op})

    return estim_specs


train_fn = lambda: DataSetIterator.get_label_features(train_filenames, 10)
test_fn = lambda: DataSetIterator.get_label_features(test_filenames, 10)

model = tf.estimator.Estimator(model_fn)
model.train(train_fn)

e = model.evaluate(test_fn)

print('auc: ', e['auc'])