from train import RecursiveNet, traverse, PredictNet, train_data_path, read_corpus
import chainer.computational_graph as c
import chainer
import json
import chainer.functions as F

sentence = '[["How @ 0", [["can @ 468", ["I @ 33", ["increase @ 222", [[["the @ 2", "speed @ 1002"], ["of @ 76", ["my @ 106", ["internet @ 1174", "connection @ 3430"]]]], ["while @ 1133", ["using @ 575", ["a @ 93", "VPN @ 2615"]]]]]]], "? @ 10"]], ["How @ 0", [["can @ 468", [["Internet @ 1176", "speed @ 1002"], ["be @ 121", ["increased @ 5998", [["by @ 123", "hacking @ 1776"], ["through @ 514", "DNS @ 26222"]]]]]], "? @ 10"]], "0", "2"]'

vocab_size = 153451
embedding_size = 30
learning_rate = 0.00001

predict = PredictNet(vocab_size, embedding_size)
import chainer.links as L
model = L.Classifier(predict)

pred = predict(json.loads(sentence))
print(pred, predict.id)

g = c.build_computational_graph([pred])
with open('./graph', 'w') as f:
    f.write(g.dump())

optimizer = chainer.optimizers.MomentumSGD(learning_rate)
optimizer.setup(predict)
optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))

train_data = read_corpus('%s/tree.0'%(train_data_path))
train_iter = chainer.iterators.SerialIterator(train_data, 10, shuffle=False)

batch = train_iter.next()
batch = [json.loads(line) for line in batch]

import numpy as np
def onehot(labels, dim=2):
    dim = max(max(labels)+1, dim)
    m = np.zeros((len(labels), dim))
    m[np.arange(len(labels)), labels] = 1
    return m

labels = [int(inst[2]) for inst in batch]
labels = onehot(labels)

accum_loss = 0
for tree in batch:
    x = tree
    #y = chainer.Variable(onehot([int(tree[2])]))
    y = chainer.Variable(np.array([int(tree[2])]))

    print(predict(x), y)
    loss = F.softmax_cross_entropy(predict(x), y)
    accum_loss += loss

model.cleargrads()
accum_loss.backward()
optimizer.update()
print(accum_loss.data)