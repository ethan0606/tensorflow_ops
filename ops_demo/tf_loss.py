import tensorflow as tf


# 手动计算logloss时，如果logit特别大，会出现nan
# tensorflow自带的不存在这个问题

def logloss(y_true, y_pred):
    y_pred_si = 1.0 / (1 + tf.exp(-y_pred))
    sigmoids = -y_true * tf.log(y_pred_si) - (1 - y_true) * tf.log(1 - y_pred_si)
    return tf.reduce_mean(sigmoids)


label = tf.constant(1.0)
logit = tf.placeholder(dtype=tf.float32, shape=())

loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit)
log_loss = logloss(label, logit)

w_loss = tf.nn.weighted_cross_entropy_with_logits(targets=label, logits=logit, pos_weight=2)

with tf.Session() as sess:
    print(sess.run(loss, feed_dict={logit: 10000}))
    print(sess.run(loss, feed_dict={logit: 0}))
    print(sess.run(loss, feed_dict={logit: 0.5}))
    print(sess.run(log_loss, feed_dict={logit: 10000}))
    print(sess.run(w_loss, feed_dict={logit: 0}))
