from train import *
from sklearn.cross_validation import KFold
from tensorflow.python.framework import ops

kf = KFold(df.shape[0], n_folds=10, shuffle=True)
epoch_number = 2
i = 0

logger.info('start validation')
for train_index, test_index in kf:
    train_set = df.iloc[train_index, :]
    test_set = df.iloc[test_index, :]

    ops.reset_default_graph()

    with tf.Session() as sess:
        graphs = build_graph(sess)

        train(train_set, './model/validate', epoch_number, graphs, sess)
        test(test_set, './model/validate', graphs, sess)