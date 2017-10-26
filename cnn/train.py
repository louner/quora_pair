import tensorflow as tf
import numpy as np
import pandas as pd
import json
import logging
from time import time
from sklearn.metrics import precision_score, recall_score

from embedding import vocab_file_path

handler = logging.FileHandler('./log/train.log')
handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

#np.random.seed(0)

df = pd.read_csv('/home/allen_kuo/quora_pair/cnn/data/train.csv')
#df = pd.read_csv('/home/allen_kuo/quora_pair/cnn/data/toy/train.csv').sample(frac=1)
dictionary = json.load(open('%s.json' % (vocab_file_path)))

batch_size = 10000
embedding_size = 300
kernel_height = 4
vocab_shape=(67114, 300)
kernel_size = (kernel_height, embedding_size)
kernel_number = 32
num_classes = 2
learning_rate = 0.001

def build_network(x, longest_length, W):
    # load embedding W
    embedding = tf.nn.embedding_lookup(params=W, ids=x)

    # reshape to NHWC
    # H -> longest sentence length
    reshape = tf.reshape(embedding, shape=(-1, longest_length, embedding_size, 1))

    # pad at both ends of a sentence
    pad = tf.pad(reshape, [[0, 0], [kernel_height - 1, kernel_height - 1], [0, 0], [0, 0]])

    # output: batch_size, longest_length+(kernel_height-1)*2-(kernel_hefight-1), 1, kernel_number
    # w-op
    conv1 = tf.layers.conv2d(pad,
                            filters=kernel_number,
                            kernel_size=kernel_size,
                            strides=(1, 1),
                            padding='valid',
                            activation=tf.tanh)

    # output: batch_size, longest_length, 1, kernel_number
    pool1 = tf.nn.pool(conv1,
                      window_shape=(kernel_height, 1),
                      pooling_type='AVG',
                      padding='VALID')

    # output: batch_size, 1, 1, kernel_number
    # all-op
    final_pool = tf.reduce_mean(pool1,
                                axis=1)
    
    # batch_size, kernel_number
    output = tf.reshape(final_pool, shape=
    (-1, kernel_number))
    return output

def padding(batch, longest_sentence_length):
    batch = np.array(batch)
    batch_size = batch.shape[0]
    zeros = np.zeros((batch_size, longest_sentence_length))
    for i in range(batch_size):
        zeros[i, :len(batch[i])] = batch[i]
    batch = zeros
    return batch

def to_word_id(batch):
    batch = [[dictionary[tok] for tok in str(sentence).split(' ') if tok in dictionary] for sentence in batch]
    longest_sentence_length = max([len(ids) for ids in batch])
    return batch, longest_sentence_length

def preprocess(batch):
    batch, longest_sentence_length = to_word_id(batch)
    batch = padding(batch, longest_sentence_length)
    return batch, longest_sentence_length

def build_graph(sess):
    W = tf.get_variable(name='W', shape=vocab_shape, trainable=False)

    # batch size and sentence length is unknown
    # tensor for longest sentence length of each batch
    labels = tf.placeholder(dtype=tf.int32)
    question1 = tf.placeholder(shape=(None, None), dtype=tf.int32)
    question2 = tf.placeholder(shape=(None, None), dtype=tf.int32)
    question1_longest_length = tf.placeholder(dtype=tf.int32)
    question2_longest_length = tf.placeholder(dtype=tf.int32)

    # reuse the same network parameters
    with tf.variable_scope('network'):
        network1 = build_network(question1, question1_longest_length, W)

    with tf.variable_scope('network', reuse=True):
        network2 = build_network(question2, question2_longest_length, W)

    # concat question1 and question2 in batch
    fc_layer = tf.concat([network1, network2], axis=1)
    predict = tf.layers.dense(fc_layer,
                    num_classes,
                    activation=tf.nn.relu,
                    use_bias=True)

    loss = tf.reduce_sum(tf.log(tf.nn.softmax(predict)) * tf.one_hot(labels, depth=num_classes)) * -1

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    init = tf.global_variables_initializer()
    sess.run(init)

    saver = tf.train.Saver(var_list=[W])
    saver.restore(sess, './models/embed_matrix.ckpt')

    return predict, loss, train_step, question1, question1_longest_length, question2, question2_longest_length, labels

def make_batch(df, st_index, batch_size=batch_size):
    batch = df[['question1', 'question2','is_duplicate']].iloc[st_index:st_index+batch_size, :]
    q1_batch, q2_batch, y = batch['question1'].values, batch['question2'].values, batch['is_duplicate'].values

    q1_batch, q1_longest_sentence_length = preprocess(q1_batch)
    q2_batch, q2_longest_sentence_length = preprocess(q2_batch)

    return q1_batch, q1_longest_sentence_length, q2_batch, q2_longest_sentence_length, y

def evaluate(predicts, labels):
    predicts = pd.DataFrame(predicts)
    pred = (predicts[1] > predicts[0]).astype(int).values

    # skip UndefinedMetricWarning when al lprediction is 0/1
    pred[0] = 1
    precision, recall = precision_score(labels, pred), recall_score(labels, pred)
    return precision, recall

if __name__ == '__main__':
    with tf.Session() as sess:

        predict, loss, train_step, question1, question1_longest_length, question2, question2_longest_length, labels = build_graph(sess)
        saver = tf.train.Saver()

        for epoch in range(1000):
            st = time()
            all_loss = 0
            df = df.sample(frac=1)
            total_steps = df.shape[0]/batch_size+1

            for step in range(total_steps):
                try:
                    st_index = step*batch_size
                    q1_batch, q1_longest_sentence_length, q2_batch, q2_longest_sentence_length, y = make_batch(df, st_index)

                    _, l = sess.run([train_step, loss], feed_dict={question1: q1_batch,
                                                               question1_longest_length: q1_longest_sentence_length,
                                                               question2: q2_batch,
                                                               question2_longest_length: q2_longest_sentence_length,
                                                               labels: y})
                    logger.info('training %d %d %f %f'%(epoch, step, time()-st, l/batch_size))
                    all_loss += l/batch_size

                except:
                    import traceback
                    logger.error(traceback.format_exc())

            logger.info('%d epoch avg loss: %f'%(epoch, all_loss/total_steps))

            #saver.save(sess, './models/cnn_one_layer_%d.ckpt'%(epoch))

            total_precision, total_recall = 0.0, 0.0
            for step in range(total_steps):
                st_index = step*batch_size
                q1_batch, q1_longest_sentence_length, q2_batch, q2_longest_sentence_length, y = make_batch(df, st_index)
                predicts = sess.run([predict], feed_dict={question1: q1_batch,
                                                           question1_longest_length: q1_longest_sentence_length,
                                                           question2: q2_batch,
                                                           question2_longest_length: q2_longest_sentence_length,
                                                           labels: y})
                predicts = predicts[0]
                precision, recall =  evaluate(predicts, y)
                total_precision += precision
                total_recall += recall

            precision, recall = total_precision/total_steps, total_recall/total_steps
            logger.info('%dth epoch PR: %f %f'%(epoch, precision, recall))
