import tensorflow as tf
import numpy as np
import pandas as pd
import json
import logging
from time import time
from sklearn.metrics import precision_score, recall_score

from embedding import vocab_file_path, tokenize

#np.random.seed(0)

handler = logging.FileHandler('./log/train.log', mode='w')
handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

df = pd.read_csv('./data/train.csv')
#df = pd.read_csv('./data/train.csv').sample(frac=0.1)

# make 50% - 50%
positive, negative = df[df['is_duplicate'] == 1], df[df['is_duplicate'] == 0]
df = df.append(positive.sample(frac=1).iloc[:negative.shape[0]-positive.shape[0], :])

dictionary = json.load(open('%s.json' % (vocab_file_path)))

logger.info('total duplicate %d'%((df['is_duplicate'] == 1).sum()))

batch_size = 5000
embedding_size = 300
kernel_height = 4
vocab_shape = (66207, 300)
kernel_number = 32
kernel_size = (kernel_height, embedding_size)
stack_kernel_size = (kernel_height, kernel_number)
num_classes = 2
learning_rate = 0.001
epoch_number = 100
layner_number = 3

def build_network(x, longest_length, W):
    # load embedding W
    embedding = tf.nn.embedding_lookup(params=W, ids=x)

    # reshape to NHWC
    # H -> longest sentence length
    reshape = tf.reshape(embedding, shape=(-1, longest_length, embedding_size, 1))

    # pad at both ends of a sentence
    pad1 = tf.pad(reshape, [[0, 0], [kernel_height - 1, kernel_height - 1], [0, 0], [0, 0]])

    # output: batch_size, longest_length+(kernel_height-1)*2-(kernel_hefight-1), 1, kernel_number
    # w-op
    conv1 = tf.layers.conv2d(pad1,
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

    reshape1 = tf.reshape(pool1, (-1, longest_length, kernel_number, 1))

    pad2 = tf.pad(reshape1, [[0, 0], [kernel_height - 1, kernel_height - 1], [0, 0], [0, 0]])
    conv2 = tf.layers.conv2d(pad2,
                            filters=kernel_number,
                            kernel_size=stack_kernel_size,
                            strides=(1, 1),
                            padding='valid',
                            activation=tf.tanh)
    pool2 = tf.nn.pool(conv2,
                      window_shape=(kernel_height, 1),
                      pooling_type='AVG',
                      padding='VALID')
    reshape2 = tf.reshape(pool2, (-1, longest_length, kernel_number, 1))

    pad3 = tf.pad(reshape2, [[0, 0], [kernel_height - 1, kernel_height - 1], [0, 0], [0, 0]])
    conv3 = tf.layers.conv2d(pad3,
                            filters=kernel_number,
                            kernel_size=stack_kernel_size,
                            strides=(1, 1),
                            padding='valid',
                            activation=tf.tanh)
    pool3 = tf.nn.pool(conv3,
                      window_shape=(kernel_height, 1),
                      pooling_type='AVG',
                      padding='VALID')

    # output: batch_size, 1, 1, kernel_number
    # all-op
    final_pool = tf.reduce_mean(pool3,
                                axis=1)
    
    # batch_size, kernel_number
    output = tf.reshape(final_pool, shape=(-1, kernel_number))
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
    batch = [[dictionary[tok] for tok in tokenize(sentence) if tok in dictionary] for sentence in batch]
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
    labels = tf.placeholder(dtype=tf.float32)
    question1 = tf.placeholder(shape=(None, None), dtype=tf.int32)
    question2 = tf.placeholder(shape=(None, None), dtype=tf.int32)
    question1_longest_length = tf.placeholder(dtype=tf.int32)
    question2_longest_length = tf.placeholder(dtype=tf.int32)

    #label_layer = tf.one_hot(labels, depth=num_classes)

    # reuse the same network parameters
    with tf.variable_scope('network'):
        network1 = build_network(question1, question1_longest_length, W)

    with tf.variable_scope('network', reuse=True):
        network2 = build_network(question2, question2_longest_length, W)

    # concat question1 and question2 in batch
    #fc_layer = tf.concat([network1, network2], axis=1)
    predict = tf.exp(-1*tf.reduce_sum(tf.abs(network1-network2), axis=1))
    loss = tf.abs(tf.cast(predict, dtype=tf.float32)-labels)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    init = tf.global_variables_initializer()
    sess.run(init)

    saver = tf.train.Saver(var_list=[W])
    saver.restore(sess, './models/embed_matrix.ckpt')

    return predict, loss, train_step, question1, question1_longest_length, question2, question2_longest_length, labels

def make_batch(df, st_index, batch_size=batch_size):
    batch = df[['question1', 'question2','is_duplicate']].iloc[st_index:st_index+batch_size, :]
    q1_batch, q2_batch, y = batch['question1'].values, batch['question2'].values, batch['is_duplicate'].values
    #logger.info(json.dumps(batch[['question1', 'question2']].values.tolist(), indent=4))

    q1_batch, q1_longest_sentence_length = preprocess(q1_batch)
    q2_batch, q2_longest_sentence_length = preprocess(q2_batch)

    #logger.info('batch: %s %s'%(str(q1_batch.shape), str(q2_batch.shape)))
    return q1_batch, q1_longest_sentence_length, q2_batch, q2_longest_sentence_length, y

def to_pred(predicts):
    pred = (predicts >= 0.5).astype(int)
    return pred

def evaluate(predicts, labels):
    pred = to_pred(predicts)
    # skip UndefinedMetricWarning when al lprediction is 0/1
    pred[0] = 1

    precision, recall = precision_score(labels, pred), recall_score(labels, pred)
    return precision, recall

def train(df, model_filepath, epoch_number, graphs, sess, do_self_evaluation=True):
    predict, loss, train_step, question1, question1_longest_length, question2, question2_longest_length, labels = graphs

    saver = tf.train.Saver()

    for epoch in range(epoch_number):
        st = time()
        all_loss = 0
        df = df.sample(frac=1)
        total_steps = int(df.shape[0]/batch_size+1)

        for step in range(total_steps):
            try:
                st_index = step*batch_size
                q1_batch, q1_longest_sentence_length, q2_batch, q2_longest_sentence_length, y = make_batch(df, st_index)

                _, l, p = sess.run([train_step, loss, predict], feed_dict={question1: q1_batch,
                                                           question1_longest_length: q1_longest_sentence_length,
                                                           question2: q2_batch,
                                                           question2_longest_length: q2_longest_sentence_length,
                                                           labels: y})

                l = l.sum()
                logger.info('training %d %d %f %f, %d %d'%(epoch, step, time()-st, l/batch_size, p.sum(), y.sum()))
                all_loss += l/batch_size

            except KeyboardInterrupt:
                raise

            except:
                import traceback
                logger.error(traceback.format_exc())

        saver.save(sess, '%s_%d'%(model_filepath, epoch))
        logger.info('%d epoch avg loss: %f'%(epoch, all_loss/total_steps))

        if do_self_evaluation:
            test(df, graphs, sess, title='training self evaluation')

def test(df, graphs, sess, model_filepath='', title=''):
    predict, loss, train_step, question1, question1_longest_length, question2, question2_longest_length, labels = graphs

    if model_filepath:
        saver = tf.train.Saver()
        saver.restore(sess, model_filepath)

    total_steps = int(df.shape[0] / batch_size + 1)

    predicts, label = np.array([]), np.array([])
    for step in range(total_steps):
        try:
            st_index = step * batch_size
            q1_batch, q1_longest_sentence_length, q2_batch, q2_longest_sentence_length, y = make_batch(df, st_index)
            p = sess.run([predict], feed_dict={question1: q1_batch,
                                                      question1_longest_length: q1_longest_sentence_length,
                                                      question2: q2_batch,
                                                      question2_longest_length: q2_longest_sentence_length,
                                                      labels: y})

            p = p[0]
            logger.info('testing %d %d' % (p.sum(), y.sum()))
        except KeyboardInterrupt:
            raise

        except:
            import traceback
            logger.error(traceback.format_exc())
            continue
        predicts = np.concatenate((predicts, p))
        label = np.concatenate((label, y))

    precision, recall = evaluate(predicts, label)
    logger.info('%s PR: %f %f' % (title, precision, recall))


if __name__ == '__main__':
    with tf.Session() as sess:
        graphs = build_graph(sess)
        train(df, './model/3_layer', epoch_number, graphs, sess)