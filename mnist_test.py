from __future__ import print_function
import tensorflow as tf
# from tensorflow.contrib import rnn
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
import math


mnist = input_data.read_data_sets("./data/", one_hot=True)

# Parameters
learning_rate = 0.001
learning_decay = 0.5

batch_size = 128
display_step = 2
keep_prob = 1.0

M = [
    'RNN', 'RNN+Highway', 'DRNN',
    'LSTM', 'LSTM + Highway', 'DLSTM',
    'GRU', 'DGRU'
         ]

'***************************************************************'
#MODUL = M[1]
training_iters = 100000
#num_layers = 10
n_hidden = 128  # hidden layer num of features
'***************************************************************'

# Network Parameters
n_input = 28  # MNIST data input (img shape: 28*28)
n_steps = 28  # timesteps
n_classes = 10  # MNIST total classes (0-9 digits)


def run(modul_index, layer):
    MODUL = M[modul_index]
    num_layers = layer


    # tf Graph input
    x = tf.placeholder("float", [None, n_steps, n_input])
    y = tf.placeholder("float", [None, n_classes])

    lr = tf.placeholder("float", shape=[])

    # Define weights
    weights = {'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))}
    biases = {'out': tf.Variable(tf.random_normal([n_classes]))}


    def x_pre(x):
        # Permuting batch_size and n_steps
        x = tf.transpose(x, [1, 0, 2])
        # Reshaping to (n_steps*batch_size, n_input)
        x = tf.reshape(x, [-1, n_input])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.split(0, n_steps, x)

        return x

    '''
    def RNN(x, weights, biases):

        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

        x = x_pre(x)

        # Define a lstm cell with tensorflow
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

        # Get lstm cell output
        outputs, states = tf.nn.rnn(lstm_cell, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']
    '''

    def run_modul(x, w, bias, modul):
        x = x_pre(x)

        def rnn_cell(modul):
            if modul == 'RNN':
                return tf.nn.rnn_cell.my_RNN(n_hidden)
            elif modul == 'RNN+Highway':
                return tf.nn.rnn_cell.my_RNN_Highway(n_hidden)
            elif modul == 'DRNN':
                return tf.nn.rnn_cell.my_DRNN(n_hidden)

            elif modul == 'LSTM':
                return tf.nn.rnn_cell.my_LSTM(n_hidden, forget_bias=0.0, state_is_tuple=True)
            elif modul == 'LSTM + Highway':
                return tf.nn.rnn_cell.my_LSTM_Highway(n_hidden, forget_bias=0.0, state_is_tuple=True)
            elif modul == 'DLSTM':
                return tf.nn.rnn_cell.my_DLSTM(n_hidden, forget_bias=0.0, state_is_tuple=True)

            elif modul == 'GRU':
                return tf.nn.rnn_cell.my_GRU(n_hidden)
            elif modul == 'DGRU':
                return tf.nn.rnn_cell.my_DGRU(n_hidden)
            else:
                raise AssertionError

        def rnn0_cell(modul):
            if modul == 'RNN':
                return tf.nn.rnn_cell.my_RNN(n_hidden)
            elif modul == 'RNN+Highway':
                return tf.nn.rnn_cell.my0_RNN_Highway(n_hidden)
            elif modul == 'DRNN':
                return tf.nn.rnn_cell.my0_DRNN(n_hidden)

            elif modul == 'LSTM':
                return tf.nn.rnn_cell.my_LSTM(n_hidden, forget_bias=0.0, state_is_tuple=True)
            elif modul == 'LSTM + Highway':
                return tf.nn.rnn_cell.my0_LSTM_Highway(n_hidden, forget_bias=0.0, state_is_tuple=True)
            elif modul == 'DLSTM':
                return tf.nn.rnn_cell.my0_DLSTM(n_hidden, forget_bias=0.0, state_is_tuple=True)

            elif modul == 'GRU':
                return tf.nn.rnn_cell.my_GRU(n_hidden)
            elif modul == 'DGRU':
                return tf.nn.rnn_cell.my0_DGRU(n_hidden)
            else:
                raise AssertionError


        attn_cell = rnn_cell
        attn0_cell = rnn0_cell

        if keep_prob < 1:
            def attn_cell(modul):
                return tf.nn.rnn_cell.DropoutWrapper(rnn_cell(modul), output_keep_prob=keep_prob)

        cell = attn0_cell(modul)
        if num_layers > 1:
            lst = [attn0_cell(modul)]
            for _ in range(num_layers - 1):
                lst.append(attn_cell(modul))
                cell = tf.nn.rnn_cell.MultiRNNCell(lst, state_is_tuple=True)

        outputs, state = tf.nn.rnn(cell, x, initial_state=cell.zero_state(batch_size, tf.float32),
                                   dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], w['out']) + bias['out']



    pred = run_modul(x, weights, biases, MODUL)

    def save_csv(dct, Acc):
        import csv
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        df = pd.DataFrame(dct)
        df_epoch = df.set_index('Epoch')
        df_epoch.plot()
        df.insert(3, 'Accuracy', Acc)
        # df = pd.concat(p)
        # print(df)
        filename = './MNIST_SAVE/mnist_' + MODUL + str(num_layers) +\
                   '*' + str(n_hidden) + '.csv'
        df.to_csv(filename, index=False)

        figname = './MNIST_SAVE/mnist_' + MODUL + str(num_layers) +\
                   '*' + str(n_hidden) + '.jpg'
        plt.savefig(figname)
        #plt.show()


    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.initialize_all_variables()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        dct = {
            'Epoch': [],
            'Loss': [],
            'Training Accuracy': []
            # 'Accuracy': []
        }

        LR = learning_rate

        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Reshape data to get 28 seq of 28 elements
            batch_x = batch_x.reshape((batch_size, n_steps, n_input))
            # Run optimization op (backprop)
            if (step * batch_size * 2) > training_iters:
                _lr = LR * learning_decay
            else:
                _lr = LR
            LR = _lr

            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, lr: _lr})
            if step % display_step == 0:
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, lr: _lr})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
                dct['Epoch'].append(int(step*batch_size))
                dct['Loss'].append(float("{:.6f}".format(loss)))
                dct['Training Accuracy'].append(float("{:.5f}".format(acc)))
            step += 1
        print("Optimization Finished!")

        # Calculate accuracy for 128 mnist test images
        test_len = 128
        test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
        test_label = mnist.test.labels[:test_len]
        Acc = sess.run(accuracy, feed_dict={x: test_data, y: test_label})

        save_csv(dct, Acc)

        print("Testing Accuracy:", Acc)
        sess.close()
    sess.close()
