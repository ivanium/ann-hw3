# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.training import moving_averages


class Model:
    def __init__(self,
                 is_train,
                 learning_rate=0.001,
                 learning_rate_decay_factor=0.9995):
        self.x_ = tf.placeholder(tf.float32, [None, 1, 28, 28])
        self.y_ = tf.placeholder(tf.int32, [None])
        self.keep_prob = tf.placeholder(tf.float32)

        x = tf.reshape(self.x_, [-1, 28, 28, 1])

        # TODO: implement input -- Conv -- BN -- ReLU -- MaxPool -- Conv -- BN -- ReLU -- MaxPool -- Linear -- loss
        W_conv1 = weight_variable([5, 5, 1, 2])
        b_conv1 = bias_variable([2])
        bnl1 =    batch_normalization_layer(conv2d(x, W_conv1) + b_conv1, is_train, '1')
        h_conv1 = tf.nn.relu(bnl1)
        h_pool1 = max_pool2x2(h_conv1)

        W_conv2 = weight_variable([5, 5, 2, 4])
        b_conv2 = bias_variable([4])
        bnl2    = batch_normalization_layer(conv2d(h_pool1, W_conv2) + b_conv2, is_train, '2')
        h_conv2 = tf.nn.relu(bnl2)
        h_pool2 = max_pool2x2(h_conv2)

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*4])
        W_linear = weight_variable([7*7*4, 10])
        b_linear = bias_variable([10])
        logits = tf.matmul(h_pool2_flat, W_linear) + b_linear
        #        the 10-class prediction output is named as "logits"
        # logits = tf.Variable(tf.constant(0.0, shape=[100, 10]))  # deleted this line after you implement above layers

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=logits))
        self.correct_pred = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), self.y_)
        self.pred = tf.argmax(logits, 1)
        self.acc = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        tf.summary.scalar('train loss', self.loss)
        tf.summary.scalar('train acc', self.acc)

        self.learning_rate = tf.Variable(float(learning_rate), trainable=False,
                                         dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)

        self.global_step = tf.Variable(0, trainable=False)
        self.params = tf.trainable_variables()
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step,
                                                                            var_list=self.params)

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2,
                                    max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)


def weight_variable(shape):  # you can use this func to build new variables
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):  # you can use this func to build new variables
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def batch_normalization_layer(inputs, isTrain=True, name=''):
    # TODO: implemented the batch normalization func and applied it on conv and fully-connected layers
    # hint: you can add extra parameters (e.g., shape) if necessary
    epsilon = 0.0001
    inputs_shape = inputs.get_shape()
    channel_shape = [int(inputs_shape[-1])]
    axis = list(range(len(inputs_shape) - 1))

    gamma = weight_variable(channel_shape)
    beta = bias_variable(channel_shape)

    epoch_mean = tf.get_variable('epoch_mean'+name, shape=channel_shape, dtype=tf.float32, 
        initializer=tf.zeros_initializer, trainable=False)
    epoch_var = tf.get_variable('epoch_var'+name, shape=channel_shape, dtype=tf.float32, 
        initializer=tf.zeros_initializer, trainable=False)
    epoch_size = tf.get_variable('epoch_size'+name, shape=[], dtype=tf.float32, 
        initializer=tf.zeros_initializer, trainable=False)


    mean, var = tf.nn.moments(inputs, axis)
    update_mean2 = tf.assign_add(epoch_mean, mean)
    update_var2 = tf.assign_add(epoch_var, var)
    update_size2 = tf.assign_add(epoch_size, tf.constant(1.))
    update_mean = moving_averages.assign_moving_average(epoch_mean, mean, 0.9997)
    update_var = moving_averages.assign_moving_average(epoch_var, var, 0.9997)

    if(not isTrain):
        # update_mean = tf.assign_add(epoch_mean, mean)
        # update_var = tf.assign_add(epoch_var, var)
        # update_size = tf.assign_add(epoch_size, tf.constant(1.))
        
        mean = update_mean2 / update_size2
        var = update_var2 / update_size2
        # mean = update_mean
        # var = update_var
        # mean = mean
        # var = var - var
    
    return tf.nn.batch_normalization(inputs, mean, var, beta, gamma, epsilon)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

def max_pool2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")