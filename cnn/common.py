import numpy as np
import logging
from embedding import vocab_file_path, tokenize
import pandas as pd
import json
from sklearn.metrics import precision_score, recall_score

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

def make_batch(df, st_index, batch_size=batch_size, is_test=False):
    batch = df.iloc[st_index:st_index+batch_size, :]
    q1_batch, q2_batch = batch['question1'].values, batch['question2'].values
    #logger.info(json.dumps(batch[['question1', 'question2']].values.tolist(), indent=4))

    q1_batch, q1_longest_sentence_length = preprocess(q1_batch)
    q2_batch, q2_longest_sentence_length = preprocess(q2_batch)

    #logger.info('batch: %s %s'%(str(q1_batch.shape), str(q2_batch.shape)))
    if not is_test:
        last = batch['is_duplicate'].values
    else:
        last = batch['test_id'].values

    return q1_batch, q1_longest_sentence_length, q2_batch, q2_longest_sentence_length, last

def to_pred(predicts):
    pred = (predicts >= 0.5).astype(int)
    return pred

def evaluate(predicts, labels):
    pred = to_pred(predicts)
    # skip UndefinedMetricWarning when al lprediction is 0/1
    pred[0] = 1

    precision, recall = precision_score(labels, pred), recall_score(labels, pred)
    return precision, recall
