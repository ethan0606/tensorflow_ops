import os
import tensorflow as tf


class FileExtractor:
    @staticmethod
    def get_filenames(directory):
        res = []
        if directory.startswith('hdfs://'):
            filenames = [f.split()[-1] for f in os.popen('hadoop fs -ls ' + directory)]
        else:
            filenames = [os.path.join(directory, f) for f in os.listdir(directory)]
        for f in filenames:
            if 'part' in f and not f.endswith("_COPYING_") and not f.endswith("SUCCESS"):
                res.append(f)
        return res


def parse_tf_example(example_proto, feature_map=None):
    if feature_map is None:
        feature_map = {
            'label': tf.io.FixedLenFeature(shape=[1], dtype=tf.float32),
            'features': tf.io.FixedLenFeature(shape=[126], dtype=tf.float32)
        }
    features = tf.parse_example(serialized=example_proto, features=feature_map)
    labels = features.pop('label')
    return features, labels


class DataSetIterator:
    @staticmethod
    def get_iter(filenames, batch=10, feature_map=None):

        if feature_map is None:
            feature_map = {
                'label': tf.io.FixedLenFeature(shape=[1], dtype=tf.float32),
                'features': tf.io.FixedLenFeature(shape=[126], dtype=tf.float32)
            }

        ds = tf.data.TFRecordDataset(filenames=filenames)
        ds = ds.batch(batch)
        ds = ds.shuffle(batch)
        ds = ds.map(lambda x: tf.io.parse_example(serialized=x, features=feature_map))
        _iter = ds.make_one_shot_iterator().get_next()
        return _iter

    @staticmethod
    def get_label_features(filenames, batch=10):
        ds = tf.data.TFRecordDataset(filenames=filenames)
        ds = ds.batch(batch)
        ds = ds.shuffle(batch)
        ds = ds.map(lambda x: parse_tf_example(x))
        features, label = ds.make_one_shot_iterator().get_next()
        return features, label



