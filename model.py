import tensorflow as tf
import numpy as np
slim = tf.contrib.slim

def model1(x):
    with tf.variable_scope('model1', values=[x], reuse=tf.AUTO_REUSE):
        with slim.arg_scope(
                [slim.conv2d],
                weights_initializer=slim.variance_scaling_initializer(),
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                stride=1, padding='SAME'):
            x = slim.conv2d(x, 32, [3, 3])
            x = slim.conv2d(x, 32, [3, 3])
            x = slim.max_pool2d(x, [2, 2], stride=[2, 2])
            x = slim.conv2d(x, 64, [3, 3])
            x = slim.conv2d(x, 64, [3, 3])
            x = slim.max_pool2d(x, [2, 2], stride=[2, 2])
            shape = x.get_shape().as_list()
            x = tf.reshape(x, [-1, np.prod(shape[1:])])
            x = slim.fully_connected(x, 200, activation_fn=tf.nn.relu)
            logits = slim.fully_connected(x, 10, activation_fn=None)
            output = tf.nn.softmax(logits)
            return logits, output


def model2(x):
    with tf.variable_scope('model2', values=[x], reuse=tf.AUTO_REUSE):
        with slim.arg_scope(
                [slim.conv2d],
                weights_initializer=slim.variance_scaling_initializer(),
                activation_fn=tf.nn.tanh,
                normalizer_fn=slim.batch_norm,
                stride=1, padding='SAME'):
            x = slim.conv2d(x, 32, [3, 3])
            x = slim.max_pool2d(x, [2, 2], stride=[2, 2])
            shape = x.get_shape().as_list()
            x = tf.reshape(x, [-1, np.prod(shape[1:])])
            x = slim.fully_connected(x, 32, activation_fn=tf.nn.tanh)
            logits = slim.fully_connected(x, 10, activation_fn=None)
            output = tf.nn.softmax(logits)
            return logits, output


def model3(x):
    with tf.variable_scope('model3', values=[x], reuse=tf.AUTO_REUSE):
        with slim.arg_scope(
                [slim.conv2d],
                weights_initializer=slim.variance_scaling_initializer(),
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                padding='SAME'):
            x = slim.conv2d(x, 32, [3, 3])
            x = slim.max_pool2d(x, [2, 2], stride=[2, 2])
            x = slim.conv2d(x, 32, [3, 3])
            x = slim.max_pool2d(x, [2, 2], stride=[2, 2])
            shape = x.get_shape().as_list()
            x = tf.reshape(x, [-1, np.prod(shape[1:])])
            x = slim.fully_connected(x, 128, activation_fn=tf.nn.relu)
            x = slim.dropout(x, 0.5)
            logits = slim.fully_connected(x, 10, activation_fn=None)
            output = tf.nn.softmax(logits)
            return logits, output


def model4(x):
    with tf.variable_scope('model4', values=[x], reuse=tf.AUTO_REUSE):
        with slim.arg_scope(
                [slim.conv2d],
                weights_initializer=slim.variance_scaling_initializer(),
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                stride=1, padding='SAME'):
            x = slim.conv2d(x, 64, [3, 3])
            x = slim.max_pool2d(x, [2, 2], stride=[2, 2])
            x = slim.conv2d(x, 128, [3, 3])
            x = slim.max_pool2d(x, [2, 2], stride=[2, 2])
            shape = x.get_shape().as_list()
            x = tf.reshape(x, [-1, np.prod(shape[1:])])
            x = slim.fully_connected(x, 128, activation_fn=tf.nn.relu)
            x = slim.fully_connected(x, 64, activation_fn=tf.nn.relu)
            x = slim.dropout(x, 0.5)
            logits = slim.fully_connected(x, 10, activation_fn=None)
            output = tf.nn.softmax(logits)
            return logits, output


def model5(x):
    with tf.variable_scope('model5', values=[x], reuse=tf.AUTO_REUSE):
        with slim.arg_scope(
                [slim.conv2d],
                weights_initializer=slim.variance_scaling_initializer(),
                activation_fn=tf.nn.tanh,
                normalizer_fn=slim.batch_norm,
                stride=1, padding='SAME'):
            x = slim.conv2d(x, 32, [3, 3])
            x = slim.max_pool2d(x, [2, 2], stride=[2, 2])
            x = slim.conv2d(x, 64, [3, 3])
            x = slim.max_pool2d(x, [2, 2], stride=[2, 2])
            shape = x.get_shape().as_list()
            x = tf.reshape(x, [-1, np.prod(shape[1:])])
            x = slim.fully_connected(x, 128, activation_fn=tf.nn.tanh)
            x = slim.dropout(x, 0.5)
            logits = slim.fully_connected(x, 10, activation_fn=None)
            output = tf.nn.softmax(logits)
            return logits, output


def model6(x):
    with tf.variable_scope('model6', values=[x], reuse=tf.AUTO_REUSE):
        with slim.arg_scope(
                [slim.conv2d],
                weights_initializer=slim.variance_scaling_initializer(),
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                padding='SAME'):
            x = slim.conv2d(x, 32, [3, 3])
            x = slim.max_pool2d(x, [2, 2], stride=[2, 2])
            x = slim.conv2d(x, 64, [3, 3])
            x = slim.max_pool2d(x, [2, 2], stride=[2, 2])
            x = slim.conv2d(x, 128, [3, 3])
            shape = x.get_shape().as_list()
            x = tf.reshape(x, [-1, np.prod(shape[1:])])
            x = slim.fully_connected(x, 128, activation_fn=tf.nn.relu)
            x = slim.dropout(x, 0.5)
            logits = slim.fully_connected(x, 10, activation_fn=None)
            output = tf.nn.softmax(logits)
            return logits, output


def model7(x):
    with tf.variable_scope('model7', values=[x], reuse=tf.AUTO_REUSE):
        with slim.arg_scope(
                [slim.conv2d],
                weights_initializer=slim.variance_scaling_initializer(),
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm):
            x = slim.conv2d(x, 32, [3, 3])
            x = slim.conv2d(x, 32, [3, 3])
            x = slim.max_pool2d(x, [2, 2], stride=[2, 2])
            x = slim.conv2d(x, 64, [3, 3])
            x = slim.conv2d(x, 128, [3, 3])
            x = slim.max_pool2d(x, [2, 2], stride=[2, 2])
            shape = x.get_shape().as_list()
            x = tf.reshape(x, [-1, np.prod(shape[1:])])
            x = slim.fully_connected(x, 128, activation_fn=tf.nn.relu)
            x = slim.dropout(x, 0.5)
            logits = slim.fully_connected(x, 10, activation_fn=None)
            output = tf.nn.softmax(logits)
            return logits, output


def model8(x):
    with tf.variable_scope('model8', values=[x], reuse=tf.AUTO_REUSE):
        shape = x.get_shape().as_list()
        x = tf.reshape(x, [-1, np.prod(shape[1:])])
        x = slim.fully_connected(x, 512, activation_fn=tf.nn.tanh)
        x = slim.dropout(x, 0.5)
        x = slim.fully_connected(x, 512, activation_fn=tf.nn.tanh)
        x = slim.dropout(x, 0.5)
        logits = slim.fully_connected(x, 10, activation_fn=None)
        output = tf.nn.softmax(logits)
        return logits, output


def model9(x):
    with tf.variable_scope('model9', values=[x], reuse=tf.AUTO_REUSE):
        shape = x.get_shape().as_list()
        x = tf.reshape(x, [-1, np.prod(shape[1:])])
        x = slim.fully_connected(x, 256, activation_fn=tf.nn.relu)
        x = slim.dropout(x, 0.5)
        x = slim.fully_connected(x, 256, activation_fn=tf.nn.relu)
        x = slim.dropout(x, 0.5)
        logits = slim.fully_connected(x, 10, activation_fn=None)
        output = tf.nn.softmax(logits)
        return logits, output


def model10(x):
    with tf.variable_scope('model10', values=[x], reuse=tf.AUTO_REUSE):
        with slim.arg_scope(
                [slim.fully_connected],
                normalizer_fn=slim.batch_norm):
            shape = x.get_shape().as_list()
            x = tf.reshape(x, [-1, np.prod(shape[1:])])
            x = slim.fully_connected(x, 512, activation_fn=tf.nn.tanh, normalizer_fn=slim.batch_norm)
            logits = slim.fully_connected(x, 10, activation_fn=None)
            output = tf.nn.softmax(logits)
            return logits, output


def model11(x):
    with tf.variable_scope('model11', values=[x], reuse=tf.AUTO_REUSE):
        with slim.arg_scope(
                [slim.fully_connected],
                normalizer_fn=slim.batch_norm):
            shape = x.get_shape().as_list()
            x = tf.reshape(x, [-1, np.prod(shape[1:])])
            x = slim.fully_connected(x, 256, activation_fn=tf.nn.relu)
            x = slim.dropout(x, 0.5)
            x = slim.fully_connected(x, 256, activation_fn=tf.nn.relu)
            x = slim.dropout(x, 0.5)
            logits = slim.fully_connected(x, 10, activation_fn=None)
            output = tf.nn.softmax(logits)
            return logits, output

def model12(x):
    with tf.variable_scope('model12', values=[x], reuse=tf.AUTO_REUSE):
        with slim.arg_scope(
                [slim.conv2d],
                weights_initializer=slim.variance_scaling_initializer(),
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                padding='SAME'):
            x = slim.conv2d(x, 64, [3, 3])
            x = slim.max_pool2d(x, [2, 2], stride=[2, 2])
            x = slim.conv2d(x, 128, [1, 1])
            x = slim.max_pool2d(x, [2, 2], stride=[2, 2])
            x = slim.conv2d(x, 64, [3, 3])
            shape = x.get_shape().as_list()
            x = tf.reshape(x, [-1, np.prod(shape[1:])])
            x = slim.fully_connected(x, 256, activation_fn=tf.nn.relu)
            x = slim.dropout(x, 0.5)
            logits = slim.fully_connected(x, 10, activation_fn=None)
            output = tf.nn.softmax(logits)
            return logits, output


def model13(x):
    with tf.variable_scope('model13', values=[x], reuse=tf.AUTO_REUSE):
        with slim.arg_scope(
                [slim.conv2d],
                weights_initializer=slim.variance_scaling_initializer(),
                activation_fn=tf.nn.tanh,
                normalizer_fn=slim.batch_norm,
                padding='SAME'):
            x = slim.conv2d(x, 32, [3, 3])
            x = slim.max_pool2d(x, [2, 2], stride=[2, 2])
            x = slim.conv2d(x, 64, [3, 3])
            x = slim.max_pool2d(x, [2, 2], stride=[2, 2])
            x = slim.conv2d(x, 128, [3, 3])
            shape = x.get_shape().as_list()
            x = tf.reshape(x, [-1, np.prod(shape[1:])])
            x = slim.fully_connected(x, 128, activation_fn=tf.nn.tanh)
            x = slim.dropout(x, 0.5)
            logits = slim.fully_connected(x, 10, activation_fn=None)
            output = tf.nn.softmax(logits)
            return logits, output

def model14(x):
    with tf.variable_scope('model14', values=[x], reuse=tf.AUTO_REUSE):
        shape = x.get_shape().as_list()
        x = tf.reshape(x, [-1, np.prod(shape[1:])])
        x = slim.fully_connected(x, 1024, activation_fn=tf.nn.tanh)
        x = slim.dropout(x, 0.5)
        x = slim.fully_connected(x, 1024, activation_fn=tf.nn.tanh)
        x = slim.dropout(x, 0.5)
        logits = slim.fully_connected(x, 10, activation_fn=None)
        output = tf.nn.softmax(logits)
        return logits, output