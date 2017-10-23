import tensorflow as tf
import numpy as np

def get_variable():
    w = tf.get_variable('w', shape=(3, 3), initializer=tf.random_normal_initializer())
    sum = tf.reduce_sum(w)
    return sum

def test(reuse=False):
    with tf.variable_scope('foo') as scope:
        sum1 = get_variable()

    with tf.variable_scope('foo', reuse=reuse) as scope:
        sum2 = get_variable()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run([sum1, sum2]))

test(True)