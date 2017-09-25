import tensorflow as tf
import numpy as np
from form_parse_tree import embedding_size, WordNode
from preprocess import load_data
import logging
import tensorflow_fold as td
import traceback
from read_data import files_reader
import json
from time import time

logging.basicConfig(filename='log/lab.log', level=logging.DEBUG, format='%(levelname)s - %(message)s')
logger = logging.getLogger('lab')

weights = {}
similarity_w = tf.Variable(tf.random_normal([2, embedding_size*2]), name='simi_w')
similarity_b = tf.Variable(tf.random_normal([2, 1]), name='simi_b')

answers = [tf.constant(np.asarray([[1], [0]]), dtype=tf.float32), tf.constant(np.asarray([[0], [1]]), dtype=tf.float32)]
learning_rate = 0.001
epsilon = tf.constant(value=1e-5)
number_classes = 2

embedding = td.Embedding(152800, embedding_size, name='embedding_layer')
fc = td.FC(embedding_size, name='fully_connected_layer')

def buid_sentence_expression():
    sentence_tree = td.InputTransform(lambda sentence_json: WordNode(sentence_json))

    tree_rnn = td.ForwardDeclaration(td.PyObjectType())
    leaf_case = td.GetItem('word_id', name='leaf_in') >> td.Scalar(dtype=tf.int32) >> embedding
    index_case = td.Record({'left': tree_rnn(), 'right': tree_rnn()}) \
                 >> td.Concat(name='concat_root_child') \
                 >> fc

    expr_sentence = td.OneOf(td.GetItem('leaf'), {True: leaf_case, False: index_case}, name='recur_in')
    tree_rnn.resolve_to(expr_sentence)

    return sentence_tree >> expr_sentence

def create_compiler():
    expr_left_sentence, expr_right_sentence = buid_sentence_expression(), buid_sentence_expression()

    expr_label = td.InputTransform(lambda label: int(label)) >> td.OneHot(2, dtype=tf.float32)
    id = td.Scalar(dtype=tf.int32)
    one_record = td.InputTransform(lambda record: json.loads(record)) >> \
                 td.Record((expr_left_sentence, expr_right_sentence, expr_label, id), name='instance')

    compiler = td.Compiler().create(one_record)
    return compiler

def build_graph():
    compiler = create_compiler()
    sentence1, sentence2, label_vector, id = compiler.output_tensors

    w = tf.Variable(tf.random_normal([2*embedding_size, number_classes]), name='similarity_w')
    b = tf.Variable(tf.random_normal([1, number_classes]), name='similarity_b')
    distance = tf.concat([sentence1, sentence2], axis=1)
    logits = tf.matmul(distance, w) + b

    loss = -1 * tf.reduce_mean(tf.log(tf.nn.softmax(logits+epsilon)) * label_vector)

    global_step = tf.Variable(0, name='global_step')
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    return logits, loss, train_step, w, global_step, compiler, id

def training():
    with tf.Session() as sess:
        file_queue = ['train/tree.%d' % (i) for i in range(10)]
        file_queue = ['data/test/tree.0']

        logits, loss, train_step, w, global_step, compiler, id = build_graph()

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        batch = compiler.build_feed_dict(files_reader(file_queue))
        st = time()
        while True:
            l, _, step, log = sess.run([loss, train_step, global_step, logits], feed_dict=batch)
            print(time()-st, l, step, len(log))
            st = time()
            saver.save(sess, './log/train/model.ckpt', global_step=global_step)

def validate():
    with tf.Session() as sess:
        file_queue = ['train/tree.%d' % (i) for i in range(10)]

        logits, loss, train_step, w, global_step, compiler, id = build_graph()

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        st = time()
        for i in range(10):
            train_set, validate_set = [file_queue[q] for q in range(len(file_queue)) if q != i], [file_queue[i]]
            traiin_batch = compiler.build_feed_dict(files_reader(train_set))
            validate_batch = compiler.build_feed_dict(files_reader(validate_set))
            l, _, step, log = sess.run([loss, train_step, global_step, logits], feed_dict=traiin_batch)
            val_loss = sess.run(loss, feed_dict=validate_batch)
            print(time() - st, l, val_loss, step, len(log))
            st = time()


if __name__ == '__main__':
    with tf.Session() as sess:
        file_queue = ['train/tree.%d' % (i) for i in range(10)]
        #file_queue = ['test/tree.%d' % (i) for i in range(10)]
        #validate_file_queue = [file_queue.pop(3)]

        logits, loss, train_step, w, global_step, batch, compiler, id = build_graph(file_queue)
        #validate_batch = compiler.build_feed_dict(files_reader(validate_file_queue))

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        #saver.restore(sess, './log/train/model.ckpt-80')

        for l, i in zip(log, idd):
            l = [str(x) for x in l.tolist()]
            print(i, ' '.join(l))

        while True:
            l, _, step, log, idd = sess.run([loss, train_step, global_step, logits, id], feed_dict=batch)
            print(time()-st, l, step, len(log))
            st = time()
            saver.save(sess, './log/train/model.ckpt', global_step=global_step)
            #l = sess.run([loss], feed_dict=validate_batch)

            #print(time()-st, l)
