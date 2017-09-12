import tensorflow as tf


class Model(object):
    def __init__(self, model, batch_size, step, learning_rate, learning_decay, rnn_size,
                 gradient_norm, net_layer, vocab_size, datatype):
        # you can read more about these models on main.py
        if model == 'simple lstm':
            hidden_cell = tf.nn.rnn_cell.BasicLSTMCell
        elif model == 'lstm':
            hidden_cell = tf.nn.rnn_cell.LSTMCell
        elif model == 'gru':
            hidden_cell = tf.nn.rnn_cell.GRUCell
        elif model == 'simple classic rnn':
            hidden_cell = tf.nn.rnn_cell.BasicRNNCell
        elif model == 'classic rnn':
            hidden_cell = tf.nn.rnn_cell.RNNCell
        else:
            raise Exception("model type not supported: " + model)

        rnn_cell = hidden_cell(rnn_size)
        # initialise size of our nets
        # number of cells in a layer times with number of layer
        with tf.variable_scope('rnnmemory'):
            self.rnn_cell = rnn_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell] * net_layer)

            self.output_data = self.input_data = tf.placeholder(datatype, [batch_size, step])
            # initial state of our nets
            self.initial_state = rnn_cell.zero_state(batch_size, tf.float32)
            self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
            self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_decay)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            self.increment_global_step_op = tf.assign_add(self.global_step, 1)

            # build a large variable

            w = tf.get_variable("proj_w", [rnn_size, vocab_size])
            b = tf.get_variable("proj_b", [vocab_size])
            embedding = tf.get_variable("embedding", [vocab_size, rnn_size])
            inputs = tf.split(1, step, tf.nn.embedding_lookup(embedding, self.input_data))
            inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

            self.outputs, self.final_state = tf.nn.seq2seq.rnn_decoder(inputs,
                                                                       self.initial_state, self.rnn_cell)
            output = tf.reshape(self.outputs, [-1, rnn_size])
            self.logits = tf.matmul(output, w) + b
            self.probs = tf.nn.softmax(self.logits)
            loss = tf.nn.seq2seq.sequence_loss_by_example([self.logits],
                                                          [tf.reshape(self.output_data, [-1])],
                                                          [tf.ones([batch_size * step])],
                                                          vocab_size)
            self.cost = tf.reduce_sum(loss) / batch_size / step
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.learning_rate_decay_op)
            optimizer = tf.train.AdagradOptimizer(self.learning_rate)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))
