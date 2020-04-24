import tensorflow as tf

ds = tf.data.Dataset.range(0, 10)
ds = ds.batch(2)
ds = ds.repeat(2)
ds = ds.map(lambda x: x-1)
ds = ds.shuffle(2)
ds = ds.make_one_shot_iterator().get_next()
with tf.Session() as sess:
    while True:
        try:
            print(sess.run(ds))
        except:
            print('iterated all data')
            break
