# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.training import moving_averages


class Model:
    def __init__(self,
                 is_train,
                 learning_rate=0.001,
                 learning_rate_decay_factor=0.9995):
        self.x_ = tf.placeholder(tf.float32, [None, 28*28])
        self.y_ = tf.placeholder(tf.int32, [None])
        self.keep_prob = tf.placeholder(tf.float32)

        # TODO:  implement input -- Linear -- BN -- ReLU -- Linear -- loss
        X = self.x_
        W1 = weight_variable([28*28, 100])
        b1 = bias_variable([100])
        bnl1 = batch_normalization_layer(tf.matmul(X, W1) + b1, is_train)
        h1 = tf.nn.relu(bnl1)
        W_l = weight_variable([100, 10])
        b_l = bias_variable([10])
        logits = tf.matmul(h1, W_l) + b_l
        #        the 10-class prediction output is named as "logits"
        # logits = tf.Variable(tf.constant(0.0, shape=[100, 10]))  # deleted this line after you implement above layers

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=logits))
        self.correct_pred = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), self.y_)
        self.pred = tf.argmax(logits, 1)  # Calculate the prediction result
        self.acc = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))  # Calculate the accuracy in this mini-batch

        tf.summary.scalar('train loss', self.loss)
        tf.summary.scalar('train acc', self.acc)

        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)  # Learning rate decay

        self.global_step = tf.Variable(0, trainable=False)
        self.params = tf.trainable_variables()
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step,
                                                                            var_list=self.params)  # Use Adam Optimizer

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2,
                                    max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)


def weight_variable(shape):  # you can use this func to build new variables
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):  # you can use this func to build new variables
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def batch_normalization_layer(inputs, isTrain=True):
    # TODO: implemented the batch normalization func and applied it on fully-connected layers
    epsilon = 0.0001
    inputs_shape = inputs.get_shape()
    length_shape = [int(inputs_shape[-1])]
    axis = list(range(len(inputs_shape) - 1))

    gamma = weight_variable(length_shape)
    beta = bias_variable(length_shape)
    
    epoch_mean = tf.get_variable('epoch_mean', shape=length_shape, dtype=tf.float32, 
        initializer=tf.zeros_initializer, trainable=False)
    epoch_var = tf.get_variable('epoch_var', shape=length_shape, dtype=tf.float32, 
        initializer=tf.zeros_initializer, trainable=False)
    epoch_size = tf.get_variable('epoch_size', shape=[], dtype=tf.float32, 
        initializer=tf.zeros_initializer, trainable=False)

    mean, var = tf.nn.moments(inputs, axis)
    update_mean2 = tf.assign_add(epoch_mean, mean)
    update_var2 = tf.assign_add(epoch_var, var)
    update_size2 = tf.assign_add(epoch_size, tf.constant(1.))
    update_mean = moving_averages.assign_moving_average(epoch_mean, mean, 0.9997)
    update_var = moving_averages.assign_moving_average(epoch_var, var, 0.9997)
    if (not isTrain):
        # update_mean = tf.assign_add(epoch_mean, mean)
        # update_var = tf.assign_add(epoch_var, var)
        # update_size = tf.assign_add(epoch_size, tf.constant(1.))
        
        mean = update_mean2 / update_size2
        var = update_var2 / update_size2
        # mean = update_mean
        # var = update_var
        # mean = mean - mean
    
    return tf.nn.batch_normalization(inputs, mean, var, beta, gamma, epsilon)

