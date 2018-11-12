# load train.py
# coding: utf-8

# In[1]:

from nltk import word_tokenize
import pandas as pd
from collections import Counter
import numpy as np
import tensorflow as tf
from time import time
from sklearn.model_selection import train_test_split
import json
from IPython import display
import matplotlib.pyplot as plt


# In[2]:

batch_size = 128*2
hidden_size = 32
embedding_size = 128
learning_rate = 100.0
reg_loss_coef = 0
dropout_keep_probability = 1.0
rnn_n_layers = 4
UNSEEN = '@unseen@'
vocab_size = 10


# In[3]:

33280/(32)


# In[4]:

np.random.seed(0)


# In[13]:

def preprocess(sentences):
    sentences = [sentence.lower() for sentence in sentences]
    return sentences

def tokenize(sentence):
    for tok in word_tokenize(sentence):
        yield tok

def make_vocab(data_filepath):
    #df = pd.read_csv(data_filepath).sample(frac=1.0)
    df = pd.read_csv(data_filepath).sample(frac=1.0).iloc[:10000]
    sentences = df['question1'].values.tolist() + df['question2'].values.tolist()
    
    sentences = preprocess(sentences)
    
    word_count = Counter([tok for sentence in sentences for tok in tokenize(sentence)])
    vocab = [word for word,count in word_count.items() if count >= 2]
    return df, vocab

def to_id(sentences, word_id):
        batch = [[word_id[tok] if tok in word_id else word_id[UNSEEN] for tok in tokenize(sentence)] for sentence in sentences]
        return batch

def pad_zero(data, length):
    val = np.zeros((length.shape[0], length.max()))
    for i,row in enumerate(data):
        llength = length[i]
        val[i, :llength] = np.asarray(row)
    return val

def to_json(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f)
        
def read_json(filepath):
    with open(filepath) as f:
        return json.load(f)


# In[14]:

def load_data(train_fpath):
    train, vocab = make_vocab('data/train.csv')
    vocab.append(UNSEEN)
    word_id = dict([(word, id+1) for id,word in enumerate(vocab)])
    word_id[UNSEEN] = 0
    vocab_size = len(vocab)

    train['question_1_id'] = to_id(train['question1'], word_id)
    train['question_2_id'] = to_id(train['question2'], word_id)
    train['question_1_length'] = train['question_1_id'].apply(lambda x: len(x))
    train['question_2_length'] = train['question_2_id'].apply(lambda x: len(x))

    return train, vocab, vocab_size



# In[ ]:

class Batch:
    def __init__(self, df, batch_size=100):
        self.df = df.sample(frac=1.0)
        self.index = 0
        self.batch_size = batch_size
        self.data = self.df[['question_1_id', 'question_2_id']]
        self.length = self.df[['question_1_length', 'question_2_length']]
        self.labels = self.df['is_duplicate']
        
        '''
        self.data['question_2_id'] = self.data['question_1_id']
        self.length['question_2_length'] = self.length['question_1_length']
        '''

    def __next__(self):
        try:
            if self.index >= self.df.shape[0]:
                raise IndexError

            next_index = self.index+self.batch_size
            if next_index > self.df.shape[0]:
                next_index = self.df.shape[0]

            data = self.data.iloc[self.index:next_index]
            label = self.labels.iloc[self.index:next_index]
            length = self.length.iloc[self.index:next_index]
            
            self.index = next_index
            
            return data, label, length
        except IndexError:
            self.__init__(self.df, self.batch_size)
            #raise StopIteration
            return self.next()

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()


# In[ ]:

class LSTM:
    def __init__(self, scope):
        self.scope = scope
        self.build()
    
    def build(self):
        self.input = tf.placeholder(shape=(None, None), dtype=tf.int32, name='%s_input'%(self.scope))
        self.batch_size = tf.shape(self.input)[0]
        self.input_length = tf.placeholder(dtype=tf.int32, name='%s_input_length'%(self.scope), shape=[None])
        self.max_input_length = tf.placeholder(dtype=tf.int32, name='%s_max_input_length'%(self.scope))
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32)
        
        self.build_embedding()
        self.build_rnn()
        
    def build_embedding(self):
        with tf.variable_scope('embed', reuse=tf.AUTO_REUSE):
            self.embedding_matrix = tf.get_variable(initializer=tf.truncated_normal(shape=[vocab_size, embedding_size]), name='embedding', trainable=False)
            self.embedding = tf.nn.embedding_lookup(params=self.embedding_matrix, ids=self.input)
            self.embedding = tf.reshape(self.embedding, shape=(-1, self.max_input_length, embedding_size))

        self.embedding = tf.nn.dropout(self.embedding, keep_prob=self.dropout_keep_prob)
    
    def build_rnn(self):
        with tf.variable_scope('lstm', reuse=tf.AUTO_REUSE):
            #cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, name='lstm_cell')
            self.cell = tf.contrib.rnn.GRUBlockCell(num_units=hidden_size)
            self.zero_state = self.cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            self.output, self.last_state = tf.nn.dynamic_rnn(cell=self.cell, inputs=self.embedding, sequence_length=self.input_length, dtype=tf.float32)


# In[12]:

class LM(LSTM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.build_loss()
        
    def build_loss(self):
        with tf.variable_scope('LM', reuse=tf.AUTO_REUSE):
            self.output = tf.gather(self.output, axis=1, indices=tf.range(self.max_input_length-1))
            self.label = tf.gather(self.input, axis=1, indices=tf.range(1, self.max_input_length))

            self.output = tf.reshape(self.output, (-1, hidden_size))
            self.logits = tf.layers.dense(inputs=self.output, units=vocab_size, activation=tf.nn.tanh, name='logits')
            self.logits = tf.reshape(self.logits, (-1, self.max_input_length-1, vocab_size))

            mask = tf.cast(tf.sequence_mask(lengths=self.input_length-1), dtype=tf.float32)
            #mask = tf.ones_like(self.label, dtype=tf.float32)
            self.loss = tf.contrib.seq2seq.sequence_loss(targets=self.label, logits=self.logits, weights=mask)
            self.tensors_shape = [tf.shape(self.logits), tf.shape(self.label), tf.shape(self.input_length), tf.shape(mask)]

def make_feed_dict(batch, lm, is_train=True):
    data, labels, lengths = batch
    lm_input_length = np.concatenate((lengths['question_1_length'].values, lengths['question_2_length'].values), axis=0)
    lm_input = pd.concat([data['question_1_id'], data['question_2_id']])
    lm_input = pad_zero(lm_input, lm_input_length)

    feed_dict = {}
    feed_dict[lm.input] = lm_input
    feed_dict[lm.input_length] = lm_input_length
    feed_dict[lm.max_input_length] = max(lm_input_length)
        
    if is_train:
        feed_dict[lm.dropout_keep_prob] = dropout_keep_probability
    else:
        feed_dict[lm.dropout_keep_prob] = 1.0
    
    return feed_dict


def train_LM():
    tf.reset_default_graph()

    global_step = tf.Variable(0, dtype=tf.int32, name='global_step', trainable=False)
    lm = LM('q')

    loss = lm.loss
    train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss, global_step=global_step)

    metrics = []
    train_data, test_data = train_test_split(train, train_size=0.8, test_size=0.2)
    train.shape, train_data.shape, test_data.shape

    with tf.Session(config=tf.ConfigProto(device_count={'GPU': 1})) as sess:
        sess.run(tf.global_variables_initializer())

        for batch in Batch(train_data, batch_size=batch_size):
            feed_dict = make_feed_dict(batch, lm)

            _, Loss, step, shape = sess.run([train_step, loss, global_step, lm.tensors_shape], feed_dict=feed_dict)

            if step % 5 == 0:
                val_loss = []
                for bt in Batch(test_data, batch_size=batch_size):
                    feed_dict = make_feed_dict(bt, lm, is_train=False)
                    val_loss.append(sess.run(loss, feed_dict=feed_dict))
                    if len(val_loss) > test_data.shape[0]/batch_size:
                        break


                metrics.append({'train_loss': Loss, 'step': step, 'val_loss': np.mean(val_loss)})
                df = pd.DataFrame(metrics)
                plt.scatter(df['step'], df['train_loss'], color='g')
                plt.scatter(df['step'], df['val_loss'], color='r')
                display.display(plt.gcf())
                display.clear_output(wait=True)

            if step >= 10000:
                break

class StackedLSTM(LSTM):
    def build_rnn(self):
        with tf.variable_scope('stacked_lstm', reuse=tf.AUTO_REUSE):
            rnn_layer = [0]*rnn_n_layers
            for i in range(rnn_n_layers):
                rnn_layer[i] = tf.contrib.rnn.GRUBlockCell(num_units=hidden_size)
                rnn_layer[i] = tf.contrib.rnn.DropoutWrapper(rnn_layer[i], output_keep_prob=dropout_keep_probability)
                
            cell = tf.contrib.rnn.MultiRNNCell(rnn_layer)
            zero_state = cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            self.output, self.last_states = tf.nn.dynamic_rnn(cell, inputs=self.embedding, sequence_length=self.input_length, initial_state=zero_state)
            self.last_state = self.last_states[-1]


# In[57]:

class OrthogonalLSM(LSTM):
    def build_rnn(self):
        with tf.variable_scope('lstm', reuse=tf.AUTO_REUSE, initializer=tf.orthogonal_initializer()):
            self.cell = tf.contrib.rnn.GRUBlockCell(num_units=hidden_size)
            self.zero_state = self.cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            self.output, self.last_state = tf.nn.dynamic_rnn(cell=self.cell, inputs=self.embedding, sequence_length=self.input_length, dtype=tf.float32)


# In[17]:

class BiLSTM(LSTM):
    def build_rnn(self):
        with tf.variable_scope('bi_lstm', reuse=tf.AUTO_REUSE):
            fw_cell = tf.contrib.rnn.GRUBlockCell(num_units=hidden_size)
            bw_cell = tf.contrib.rnn.GRUBlockCell(num_units=hidden_size)
            
            fw_zero_state = fw_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            bw_zero_state = bw_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            
            outputs, last_states = tf.nn.bidirectional_dynamic_rnn(fw_cell,
                                                                   bw_cell,
                                                                   self.embedding,
                                                                   sequence_length=self.input_length,
                                                                   dtype=tf.float32,
                                                                   initial_state_fw=fw_zero_state,
                                                                   initial_state_bw=bw_zero_state)
            
            self.last_state = tf.concat(last_states, axis=1)


def build_network():
    #l2 similairty loss
    global_step = tf.Variable(0, dtype=tf.int32, name='global_step', trainable=False)
    label = tf.placeholder(dtype=tf.float32, name='label')


    q1 = LSTM('q1')
    q2 = LSTM('q2')

    '''
    q1 = StackedLSTM('q1')
    q2 = StackedLSTM('q2')
    
    q1 = BiLSTM('q1')
    q2 = BiLSTM('q2')
    '''

    similarity = tf.exp(tf.norm(q1.last_state-q2.last_state, ord=1, axis=1)*-1)
    predict = tf.cast(similarity >= 0.5, dtype=tf.float32)
    loss = tf.nn.l2_loss(similarity-label)

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(label, predict), dtype=tf.float32))


def clip():
    #clip gradient by norm, prevent observed gradient exploding
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grad_vars = optimizer.compute_gradients(loss)
    grad_vars = [(tf.clip_by_norm(g, clip_norm=5), var) for g,var in grad_vars if var is not None]
    train_step = optimizer.apply_gradients(grads_and_vars=grad_vars, global_step=global_step)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(label, predict), dtype=tf.float32))


# In[56]:

def make_feed_dict(batch, q1, q2, label, is_train=True):
    data, labels, lengths = batch
    q1_input = pad_zero(data['question_1_id'], lengths['question_1_length'].values)
    q2_input = pad_zero(data['question_2_id'], lengths['question_2_length'].values)

    feed_dict = {}
    feed_dict[label] = labels
    feed_dict[q1.input] = q1_input
    feed_dict[q2.input] = q2_input
    feed_dict[q1.input_length] = lengths['question_1_length'].values
    feed_dict[q2.input_length] = lengths['question_2_length'].values
    feed_dict[q1.max_input_length] = q1_input.shape[1]
    feed_dict[q2.max_input_length] = q2_input.shape[1]
    
    if is_train:
        feed_dict[q1.dropout_keep_prob] = dropout_keep_probability
        feed_dict[q2.dropout_keep_prob] = dropout_keep_probability
    else:
        feed_dict[q1.dropout_keep_prob] = 1.0
        feed_dict[q2.dropout_keep_prob] = 1.0
    
    return feed_dict


# In[57]:

def train():

    metrics = []
    train_data, test_data = train_test_split(train, train_size=0.8, test_size=0.2)
    train.shape, train_data.shape, test_data.shape


    # In[58]:

    with tf.Session(config=tf.ConfigProto(device_count={'GPU': 1})) as sess:
        #writer = tf.summary.FileWriter('summary/%d'%(int(time())), sess.graph)
        writer = tf.summary.FileWriter('summary', sess.graph)

        sess.run(tf.global_variables_initializer())
        for batch in Batch(train_data, batch_size=batch_size):
            feed_dict = make_feed_dict(batch, q1, q2)

            _, Loss, step, acc, gradients = sess.run([train_step, loss, global_step, accuracy, grad], feed_dict=feed_dict)

            if step % 5 == 0:
                summary = tf.Summary()
                summary.value.add(tag='train/loss', simple_value=Loss)
                writer.add_summary(summary, global_step=step)
                writer.flush()

            if step % 100 == 0:
                val_acc = []
                for bt in Batch(test_data, batch_size=batch_size):
                    feed_dict = make_feed_dict(bt, q1, q2, is_train=False)
                    val_acc.append(sess.run(accuracy, feed_dict=feed_dict))
                    if len(val_acc) > test_data.shape[0]/batch_size:
                        break

                metric = {'train_loss': Loss, 'step': step, 'acc': acc, 'val_acc': np.mean(val_acc)}
                for i,g in enumerate(gradients):
                    metric['grad_%d'%(i)] = np.linalg.norm(g, ord=2)
                metrics.append(metric)

                df = pd.DataFrame(metrics)
                plt.scatter(df['step'], df['acc'])
                #plt.scatter(df['step'], df['train_loss'])

                '''
                plt.scatter(df['step'], df['grad_0'])
                plt.scatter(df['step'], df['grad_1'])
                plt.scatter(df['step'], df['grad_2'])
                plt.scatter(df['step'], df['grad_3'])
                '''

                plt.scatter(df['step'], df['val_acc'])
                display.display(plt.gcf())
                display.clear_output(wait=True)

                #print(step, Loss, acc)

            if step >= 10000:
                break


def plot():
    df = pd.DataFrame(metrics)

    plt.scatter(df['step'], df['grad_0'])
    plt.scatter(df['step'], df['grad_1'])
    plt.scatter(df['step'], df['grad_2'])
    plt.scatter(df['step'], df['grad_3'])


    # In[83]:

    df = pd.DataFrame(metrics)
    plt.scatter(df['step'], df['acc'])
    #plt.scatter(df['step'], df['train_loss'])
    '''
    plt.scatter(df['step'], df['grad_0'])
    plt.scatter(df['step'], df['grad_1'])
    plt.scatter(df['step'], df['grad_2'])
    plt.scatter(df['step'], df['grad_3'])
    '''

    plt.scatter(df['step'], df['val_acc'])


    # In[22]:

    print(grad_vars)


    # In[81]:

    train.shape, train_data.shape, test_data.shape


    # In[131]:

    train['is_duplicate'].mean()


    # In[ ]:



