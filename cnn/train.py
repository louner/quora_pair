import tensorflow as tf
import numpy as np
import pandas as pd
import json

#np.random.seed(0)

vocab_file_path = '/home/louner/school/ml/tree-rnn/data/vocab'
df = pd.read_csv('/home/louner/school/ml/quora_pair/cnn/data/train.csv')
question1 = df['question1']
dictionary = json.load(open('%s.json' % (vocab_file_path)))

batch_size = 500
embedding_size = 300
kernel_height = 4
vocab_shape=(85540, 300)

# batch size and sentence length is unknown
inp = tf.placeholder(shape=(None, None), dtype=tf.int32)

# tensor for longest sentence length of each batch
longest_length = tf.placeholder(dtype=tf.int32)

# load embedding W
W = tf.get_variable(name='W', shape=vocab_shape, trainable=False)
embedding = tf.nn.embedding_lookup(params=W, ids=inp)

# reshape to NHWC
# H -> longest sentence length
reshape = tf.reshape(embedding, shape=(-1, longest_length, embedding_size, 1))

# pad at both ends of a sentence
pad = tf.pad(reshape, [[0, 0], [kernel_height-1, kernel_height-1], [0, 0], [0, 0]])

conv1 = tf.layers.conv2d(pad,
                        filters=32,
                        kernel_size=(kernel_height, embedding_size),
                        strides=(1, 1),
                        padding='valid',
                        activation=tf.tanh)



with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    saver = tf.train.Saver(var_list=[W])
    saver.restore(sess, './models/embed_matrix.ckpt')

    for step in range(1):
        batch = question1.sample(batch_size).values

        batch = [[dictionary[tok] for tok in sentence.split(' ') if tok in dictionary] for sentence in batch]

        longest_sentence_length = max([len(ids) for ids in batch])
        batch = np.array(batch)
        zeros = np.zeros((batch_size, longest_sentence_length))
        for i in range(batch_size):
            zeros[i, :len(batch[i])] = batch[i]
        batch = zeros
        print(longest_sentence_length)

        d = sess.run(conv1, feed_dict={inp: batch, longest_length: longest_sentence_length})
        print(d.shape)