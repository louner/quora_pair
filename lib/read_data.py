import tensorflow as tf
import json

def read(file_queue):
    reader = tf.TextLineReader()
    key, value = reader.read(file_queue)
    return [value]

def make_batch(file_queue, batch_size=10):
    readers = [read(file_queue) for _ in range(10)]
    batch_size = batch_size
    min_after_dequeue = batch_size
    capacity = min_after_dequeue*3 + batch_size

    value_batch = tf.train.shuffle_batch_join(readers,
                                batch_size=batch_size,
                                capacity=capacity,
                                min_after_dequeue=min_after_dequeue)

    return value_batch

def get_batch(sess, filenames, num_epochs=10):
    file_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs)
    batch = make_batch(file_queue)

    sess.run(tf.local_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    return batch

def files_reader(files):
    for f in files:
        with open(f) as fin:
            for line in fin:
                yield line.strip('\n')

if __name__ == '__main__':
    with tf.Session() as sess:
        batch = get_batch(sess, ['train/tree.%d' % (i) for i in range(10)])
        print(sess.run(batch))