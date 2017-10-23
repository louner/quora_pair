import tensorflow as tf

inp = tf.constant([0, 0, 1, 1])
onehot = tf.one_hot(inp, depth=2)

with tf.Session() as sess:
    print(sess.run(onehot))