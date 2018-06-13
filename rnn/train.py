
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
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:

batch_size = 128
hidden_size = 64
embedding_size = 128
learning_rate = 0.01
UNSEEN = '@unseen@'


# In[3]:

def preprocess(sentences):
    sentences = [sentence.lower() for sentence in sentences]
    return sentences

def tokenize(sentence):
    for tok in word_tokenize(sentence):
        yield tok

def make_vocab(data_filepath):
    #df = pd.read_csv(data_filepath).sample(frac=1.0)
    df = pd.read_csv(data_filepath).sample(frac=1.0).iloc[:1000]
    sentences = df['question1'].values.tolist() + df['question2'].values.tolist()
    
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


# In[4]:

train, vocab = make_vocab('data/train.csv')
vocab.append(UNSEEN)
word_id = dict([(word, id+1) for id,word in enumerate(vocab)])
word_id[UNSEEN] = 0
vocab_size = len(vocab)


# In[5]:

vocab_size


# In[5]:

train['question_1_id'] = to_id(train['question1'], word_id)
train['question_2_id'] = to_id(train['question2'], word_id)
train['question_1_length'] = train['question_1_id'].apply(lambda x: len(x))
train['question_2_length'] = train['question_2_id'].apply(lambda x: len(x))


# In[21]:

'''
train.to_csv('data/all.csv')
to_json('data/vocab.json', [vocab, vocab_size])
'''


# In[4]:

'''
train = pd.read_csv('data/all.csv')
vocab, vocab_size = read_json('data/vocab.json')
'''


# In[6]:

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


# In[7]:

def lstm(scope):
    inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='%s_inputs'%(scope))
    inputs_length = tf.placeholder(dtype=tf.int32, name='%s_inputs_length'%(scope))
    max_inputs_length = tf.placeholder(dtype=tf.int32, name='%s_max_inputs_length'%(scope))
    
    with tf.variable_scope('embed', reuse=tf.AUTO_REUSE):
        embedding_matrix = tf.get_variable(initializer=tf.truncated_normal(shape=[vocab_size, embedding_size]), name='embedding')
        embedding = tf.nn.embedding_lookup(params=embedding_matrix, ids=inputs)
        embedding = tf.reshape(embedding, shape=(-1, max_inputs_length, embedding_size))

    with tf.variable_scope('lstm', reuse=tf.AUTO_REUSE):
        #cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, name='lstm_cell')
        cell = tf.contrib.rnn.GRUBlockCell(num_units=hidden_size)
        zero_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)
        outputs, last_state = tf.nn.dynamic_rnn(cell=cell, inputs=embedding, sequence_length=inputs_length, dtype=tf.float32)
    return inputs, inputs_length, max_inputs_length, outputs, last_state


# In[13]:

tf.reset_default_graph()


# In[10]:

#l2 similairty loss
global_step = tf.Variable(0, dtype=tf.int32, name='global_step')
label = tf.placeholder(dtype=tf.float32, name='label')

q1_input, q1_input_length, q1_max_input_length, q1_outputs, q1_last_state = lstm('q1')
q2_input, q2_input_length, q2_max_input_length, q2_outputs, q2_last_state = lstm('q2')

similarity = tf.exp(tf.norm(q1_last_state-q2_last_state, ord=1, axis=1)*-1)
predict = tf.cast(similarity >= 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(label, predict), dtype=tf.float32))
l2_loss = tf.nn.l2_loss(similarity-label)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(l2_loss, global_step=global_step)


# In[14]:

#concated l1/dot similarity & cross entropy loss
global_step = tf.Variable(0, dtype=tf.int32, name='global_step')
label = tf.placeholder(dtype=tf.int64, name='label')
label_cat = tf.one_hot(label, 2)

q1_input, q1_input_length, q1_max_input_length, q1_outputs, q1_last_state = lstm('q1')
q2_input, q2_input_length, q2_max_input_length, q2_outputs, q2_last_state = lstm('q2')

l1_similarity = tf.abs(q1_last_state-q2_last_state)
dot_similarity = tf.multiply(q1_last_state, q2_last_state)
similarity = tf.concat([l1_similarity, dot_similarity], axis=1)

logits = tf.layers.dense(inputs=similarity, units=2, activation=tf.nn.tanh)
loss = tf.nn.softmax_cross_entropy_with_logits(labels=label_cat, logits=logits)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

predict = tf.argmax(logits, axis=1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(label, predict), dtype=tf.float32))


# In[9]:

def make_feed_dict(batch):
    data, labels, lengths = batch
    q1 = pad_zero(data['question_1_id'], lengths['question_1_length'].values)
    q2 = pad_zero(data['question_2_id'], lengths['question_2_length'].values)

    feed_dict = {}
    feed_dict[label] = labels
    feed_dict[q1_input] = q1
    feed_dict[q2_input] = q2
    feed_dict[q1_input_length] = lengths['question_1_length'].values
    feed_dict[q2_input_length] = lengths['question_2_length'].values
    feed_dict[q1_max_input_length] = q1.shape[1]
    feed_dict[q2_max_input_length] = q2.shape[1]
    return feed_dict


# In[10]:

metrics = []
train_data, test_data = train_test_split(train, train_size=0.8, test_size=0.2)
train.shape, train_data.shape, test_data.shape


# In[15]:

with tf.Session(config=tf.ConfigProto(device_count={'GPU': 1})) as sess:
    #writer = tf.summary.FileWriter('summary/%d'%(int(time())), sess.graph)
    writer = tf.summary.FileWriter('summary', sess.graph)
    
    sess.run(tf.global_variables_initializer())
    for batch in Batch(train_data, batch_size=batch_size):
        feed_dict = make_feed_dict(batch)
        
        
        _, Loss, step, acc = sess.run([train_step, loss, global_step, accuracy], feed_dict=feed_dict)
                
        if step % 5 == 0:
            summary = tf.Summary()
            summary.value.add(tag='train/loss', simple_value=Loss)
            writer.add_summary(summary, global_step=step)
            writer.flush()
            print(Loss, step)

            
        if step % 100 == 0:
            val_acc = []
            for bt in Batch(test_data, batch_size=batch_size):
                feed_dict = make_feed_dict(bt)
                val_acc.append(sess.run(accuracy, feed_dict=feed_dict))
                if len(val_acc) > test_data.shape[0]/batch_size:
                    break
            
            metrics.append({'train_loss': Loss, 'step': step, 'acc': acc, 'val_acc': np.mean(val_acc)})
            
        if step >= 10000:
            break


# In[15]:

metrics = pd.DataFrame(metrics)
plt.scatter(metrics['step'], metrics['acc'])
#plt.scatter(metrics['step'], metrics['train_loss'])
plt.scatter(metrics['step'], metrics['val_acc'])
metrics.to_csv('metrics.csv')


# In[ ]:

metrics = pd.read_csv('metrics.csv')
plt.scatter(metrics['step'], metrics['train_loss'])


# In[ ]:

train_data, test_data = train_test_split(train, train_size=0.8)


# In[ ]:

train.shape, train_data.shape, test_data.shape


# In[ ]:



