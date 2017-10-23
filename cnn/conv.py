import tensorflow as tf
import pandas as pd
import numpy as np

#from embedding import batch_size
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

# 55000, 784
batch_size = 25
image_size = 784
batch_x, batch_y = mnist.train.next_batch(batch_size)

def nn_conv2d():
    # batch_size, instance_dim
    X = tf.placeholder(tf.float32, shape=[None, 784])

    # reshape to batch_size, height, width, channel == NHWC
    x = tf.reshape(X, shape=(-1, 28, 28, 1))

    # pad height with up 2 and down 2, zeros -> batch_size, 2+height+2, width, channel
    xx = tf.pad(x, paddings=[[0, 0], [2, 2], [0, 0], [0, 0]])

    # height, width, in_channel, out_channel
    filters = tf.Variable(tf.random_normal(shape=(5, 5, 1, 32)))

    # VALID: one filter scan == (28-5+1)*(28-5+1)*1, total batch_size * (28-5+1)*(28-5+1)*1 * filter_num
    # SAME: one filter scan == 28*28, total batch_size * 28*28 * filter_num
    conv1 = tf.nn.conv2d(xx, filters, strides=(1, 1, 1, 1), padding='VALID')

    # tf.nn.bias_add: [i, j, k. ..., :] += bias, for all i,j,k,...
    bias = tf.Variable(tf.random_normal(shape=[32]))
    conv1_biased = tf.nn.bias_add(conv1, bias)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        d = sess.run(conv1_biased, feed_dict={X: batch_x})
        print(d.shape)

def layers_conv():
    X = tf.placeholder(tf.float32, shape=[None, 784])
    x = tf.reshape(X, shape=(-1, 28, 28, 1))

    conv = tf.layers.conv2d(x,
                     filters=32,
                     kernel_size=(5, 5),
                     strides=(1, 1),
                     padding='valid',
                     activation=tf.tanh)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        d = sess.run(conv, feed_dict={X: batch_x})
        print(d.shape)

def pool():
    mat = np.zeros((3,3))
    mat[2, 2] = 1
    print(mat)
    inp = tf.reshape(tf.constant(mat, dtype=tf.float32), shape=(1, 3, 3, 1))
    pool = tf.nn.pool(inp, window_shape=(2, 2), pooling_type='MAX', padding='VALID')

    with tf.Session() as sess:
        d = sess.run(pool)
        print(d)

#nn_conv2d()
#layers_conv()