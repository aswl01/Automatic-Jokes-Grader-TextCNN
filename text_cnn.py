import tensorflow as tf

def get_W(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='W')


def get_b(num_filters):
    return tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')


def conv2d_Valid(x, W, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID', name='conv'), b),
                      name='relu')


class Text_CNN(object):
    def __init__(self, sentence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters,
                 l2_reg_lambda=0.0):
        self.input_x = tf.placeholder(tf.int32, [None, sentence_length], name='input_x')
        self.input_y = tf.placeholder(tf.int32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # embedding layer
        with tf.name_scope('embedding'):
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, -1.0), name='W')
            # shape of embedded_chars = [ sentence_length, embedding_size ]
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_output = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope('block-%s' % filter_size):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = get_W(filter_shape)
                b = get_b(num_filters)
                conv = conv2d_Valid(self.embedded_chars_expanded, W, b)
                pooled = tf.nn.max_pool(conv, ksize=[1, sentence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                        padding='VALID', name='pool')
                pooled_output.append(pooled)

        # Combine all the pooled features
        total_filter_size = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_output, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, total_filter_size])

        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope('output'):
            W = tf.get_variable('W', shape=[total_filter_size, num_classes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name='scores')
            self.predictions = tf.argmax(self.scores, 1, name='predictions')

        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')
