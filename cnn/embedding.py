import json
import numpy as np
import tensorflow as tf
import pandas as pd
embedding_size = 300

from gensim.models.keyedvectors import KeyedVectors
word2vec_model_filepath = '/home/louner/school/ml/word2vec-GoogleNews-vectors/GoogleNews-vectors-negative300.bin'
#w2v = KeyedVectors.load_word2vec_format(word2vec_model_filepath, binary=True)

vocab_file_path = '/home/louner/school/ml/tree-rnn/data/vocab'
vocab_shape=(85540, 300)
batch_size = 5

import logging

handler = logging.FileHandler('/home/louner/school/ml/quora_pair/cnn/log/embeddding.log', mode='w')
handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

dictionary = json.load(open('%s.json' % (vocab_file_path)))

np.random.seed(0)

def make_embed_matrix(vocab_file_path):
    embed_matrix = []
    with open(vocab_file_path) as f:
        for word in f:
            word = word.strip('\n').lower()
            try:
                vec = w2v.word_vec(word)
                embed_matrix.append(vec)
                dictionary[word] = len(dictionary)
            except:
                logger.error('UNKNOWN %s'%(word))
    embed_matrix = np.array(embed_matrix)
    logger.info('shape: %s'%str(embed_matrix.shape))

    json.dump(dictionary, open('%s.json'%(vocab_file_path), 'w'))

    return np.reshape(embed_matrix, embed_matrix.shape)

def save_embed():
    embed_matrix = make_embed_matrix('/home/louner/school/ml/tree-rnn/data/vocab')

    W = tf.get_variable(name='W', shape=embed_matrix.shape, initializer=tf.constant_initializer(embed_matrix))

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver(var_list=[W])
        saver.save(sess, './models/embed_matrix.ckpt')

def load_embed():
    W = tf.get_variable(name='W', shape=shape, trainable=False)
    ids = tf.constant([0, 1, 0])
    lookup = tf.nn.embedding_lookup(params=W, ids=ids)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver(var_list=[W])
        saver.restore(sess, './models/embed_matrix.ckpt')
        print(sess.run(lookup))

def to_ids(sentence):
    toks = sentence.decode('utf-8').split(' ')
    ids = [dictionary[tok] for tok in toks if tok in dictionary]
    return np.array(ids, dtype=np.int32)

def to_sentence_matrix(sentence):
    ids = tf.py_func(to_ids, [sentence], tf.int32)
    embedding = tf.nn.embedding_lookup(params=W, ids=ids)
    return embedding

def trans():
    W = tf.get_variable(name='W', shape=shape, trainable=False)

    sentence1 = 'What is the step by step guide to invest in share market in india?'
    sentence2 = 'What is the step by step guide to invest in share market?'
    sentence_matrix1, sentence_matrix2 = to_sentence_matrix(sentence1), to_sentence_matrix(sentence2)

    with tf.Session() as sess:
        saver = tf.train.Saver(var_list=[W])
        saver.restore(sess, './models/embed_matrix.ckpt')

        print(sess.run([sentence_matrix1, sentence_matrix2]))

def make_batch():
    df = pd.read_csv('/home/louner/school/ml/quora_pair/cnn/data/train.csv')
    sentences = df['question1']
    batch = sentences.sample(n=batch_size).values
    print(batch)
    batch = [[dictionary[tok] for tok in sentence.split(' ') if tok in dictionary] for sentence in batch]
    longest_sentence_length = max([len(ids) for ids in batch])
    print(batch)
    print(longest_sentence_length)
    #shape = tf.constant([[0, 0], [0, longest_sentence_length]])
    batch = np.array(batch)
    zeros = np.zeros((batch_size, longest_sentence_length))
    for i in range(batch_size):
        zeros[i, :len(batch[i])] = batch[i]
    batch = zeros
    print(batch.shape)

    input = tf.placeholder(shape=(None, longest_sentence_length), dtype=tf.int32)
    W = tf.get_variable(name='W', shape=vocab_shape, trainable=False)
    embedding = tf.nn.embedding_lookup(params=W, ids=input)

    with tf.Session() as sess:
        saver = tf.train.Saver(var_list=[W])
        saver.restore(sess, './models/embed_matrix.ckpt')

        print(sess.run(embedding, feed_dict={input: batch}))

#save_embed()
#load_embed()
#trans()
make_batch()