import tensorflow as tf

file_list = ['/Users/yifanguo/Desktop/tensorflow_ops/data/agaricus/tf_test']

feature_map = {
    'label': tf.io.FixedLenFeature(shape=[1], dtype=tf.float32),
    'features': tf.io.FixedLenFeature(shape=[126], dtype=tf.float32)
}

ds = tf.data.TFRecordDataset(filenames=file_list, num_parallel_reads=2)
ds = ds.batch(1)
ds = ds.map(lambda x: tf.io.parse_example(serialized=x, features=feature_map))
batch = ds.make_one_shot_iterator().get_next()

with tf.Session() as sess:
    print(sess.run(batch['label']))
