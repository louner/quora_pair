from train import *

np.random.seed(0)

model_filepath = './model/3_layer_99'

if __name__ == '__main__':
    df = pd.read_csv('./data/test.csv')

    with tf.Session() as sess:
        graphs = build_graph(sess)
        predict, loss, train_step, question1, question1_longest_length, question2, question2_longest_length, labels = graphs

        saver = tf.train.Saver()
        saver.restore(sess, model_filepath)

        total_steps = int(df.shape[0] / batch_size + 1)

        predicts, label, ids = np.array([]), np.array([]), np.array([])
        for step in range(total_steps):
            try:
                st_index = step * batch_size
                q1_batch, q1_longest_sentence_length, q2_batch, q2_longest_sentence_length, test_ids = make_batch(df, st_index, is_test=True)
                p = sess.run([predict], feed_dict={question1: q1_batch,
                                                   question1_longest_length: q1_longest_sentence_length,
                                                   question2: q2_batch,
                                                   question2_longest_length: q2_longest_sentence_length})
                p = p[0]
            except KeyboardInterrupt:
                raise

            except:
                import traceback
                logger.error(traceback.format_exc())

            predicts = np.concatenate((predicts, p))
            ids = np.concatenate((ids, test_ids))

    ids = ids.tolist()
    predicts = predicts.tolist()

    for id, pred in zip(ids, predicts):
        pred = 1 if pred >= 0.5 else 0
        print('%d,%s'%(int(id), pred))