import tensorflow as tf
import numpy as np
from common import *

from time import time
#np.random.seed(0)

embedding_size = 300
kernel_height = 4
vocab_shape = (66207, 300)
kernel_number = 32
kernel_size = (kernel_height, embedding_size)
stack_kernel_size = (kernel_height, kernel_number)
num_classes = 2
learning_rate = 0.001
epoch_number = 300
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
    reshape3 = tf.reshape(pool2, (-1, longest_length, kernel_number, 1))

    # output
    appended_outputs = tf.concat([reshape1, reshape2, reshape3], axis=2)
    # output: batch_size, 1, 1, kernel_number
    # all-op
    final_pool = tf.reduce_mean(appended_outputs, axis=1)
    
    # batch_size, kernel_number
    output = tf.reshape(final_pool, shape=(-1, kernel_number*3))
    return output

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

def train(df, model_filepath, epoch_number, graphs, sess, do_self_evaluation=True):
    predict, loss, train_step, question1, question1_longest_length, question2, question2_longest_length, labels = graphs

    saver = tf.train.Saver()

    df = df.sample(frac=1)
    test_st = int(df.shape[0]*0.8)
    train_set, test_set = df.iloc[:test_st, :], df.iloc[test_st:, :]

    for epoch in range(epoch_number):
        st = time()
        all_loss = 0
        train_set = train_set.sample(frac=1)
        total_steps = int(train_set.shape[0]/batch_size+1)

        for step in range(total_steps):
            try:
                st_index = step*batch_size
                q1_batch, q1_longest_sentence_length, q2_batch, q2_longest_sentence_length, y = make_batch(train_set, st_index)

                _, l, p = sess.run([train_step, loss, predict], feed_dict={question1: q1_batch,
                                                           question1_longest_length: q1_longest_sentence_length,
                                                           question2: q2_batch,
                                                           question2_longest_length: q2_longest_sentence_length,
                                                           labels: y})

                l = l.sum()
                #logger.info('training %d %d %f %f, %d %d'%(epoch, step, time()-st, l/batch_size, p.sum(), y.sum()))
                all_loss += l/batch_size

            except KeyboardInterrupt:
                raise

            except:
                import traceback
                logger.error(traceback.format_exc())

        saver.save(sess, '%s_%d'%(model_filepath, epoch))
        logger.info('%d epoch avg loss: %f'%(epoch, all_loss/total_steps))

        if do_self_evaluation:
            test(test_set, graphs, sess, title='training self evaluation')

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
            #logger.info('testing %d %d' % (p.sum(), y.sum()))
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
        train(df, './model/3_layer_append_outputs', epoch_number, graphs, sess)