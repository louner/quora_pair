from train import *
from sklearn.cross_validation import KFold
from tensorflow.python.framework import ops

df = df.sample(frac=1)
kf = KFold(df.shape[0], n_folds=10, shuffle=True)
epoch_number = 1
i = 0
print(df.shape)

logger.info('start validation')
step = 0
for train_index, test_index in kf:
    train_set = df.iloc[train_index, :]
    test_set = df.iloc[test_index, :]

    ops.reset_default_graph()

    with tf.Session() as sess:
        graphs = build_graph(sess)

        train(train_set, './models/validate', epoch_number, graphs, sess)
        test(test_set, './models/validate', graphs, sess, str(step))
        step += 1