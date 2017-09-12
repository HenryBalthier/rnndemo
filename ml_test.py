# -*- coding: utf8 -*-

from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

rdm = np.random


def test():
    hello = tf.constant('Hello World!')

    a = tf.placeholder(tf.int16)
    b = tf.placeholder(tf.int16)

    matrix1 = tf.constant([[3, 3]])
    matrix2 = tf.constant([[2], [2]])
    result = tf.matmul(matrix1, matrix2)

    add = tf.add(a, b)
    mul = tf.mul(a, b)

    with tf.Session() as sess:
        print(sess.run(hello))
        print(sess.run(mul + add, feed_dict={a: 2, b: 3}))
        print(sess.run(result))


def liner():
    # y = [[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]
    y = [None, 28, 28]

    y1 = tf.transpose(y, perm=[0, 1, 2])
    y2 = tf.transpose(y, perm=[0, 2, 1])
    y3 = tf.transpose(y, perm=[1, 0, 2])

    with tf.Session() as sess:
        print(sess.run(y1))
        print('-----------\n')
        print(sess.run(y2))
        print('-----------\n')
        print(sess.run(y3))
        print('-----------\n')

def fn(x, y):
    return x + y

def scan():
    elems = tf.Variable([1, 2, 2, 2, 2])
    e = tf.identity(elems)
    init = tf.constant(0)
    out = tf.scan(fn, elems, initializer=init)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        print(sess.run(out))


if __name__ == '__main__':
    scan()
