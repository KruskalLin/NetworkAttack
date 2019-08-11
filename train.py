import tensorflow as tf
import os
from config import get_dataset, get_models
import numpy as np
from model import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
max_epsilon = 25.0
num_iter = 32
momentum = 1
configs = {
    'batch_size': 64,
    'epoch': 5,
    'kernel_size': 7
}
(X_train, y_train), (X_test, y_test) = get_dataset()
to_categorical = tf.keras.utils.to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
ind = np.random.permutation(X_train.shape[0])
X_train, y_train = X_train[ind], y_train[ind]

VALIDATION_SPLIT = 0.1
n = int(X_train.shape[0] * (1-VALIDATION_SPLIT))
X_valid = X_train[n:]
X_train = X_train[:n]
y_valid = y_train[n:]
y_train = y_train[:n]

# model_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# models = get_models(model_index)


class Dummy:
    pass


env = Dummy()
print('\nConstruction graph')
env.x = tf.placeholder(tf.float32, (None, 28, 28, 1), name='x')
env.y = tf.placeholder(tf.float32, (None, 10), name='y')
env.ybar, logits = model4(env.x)
with tf.variable_scope('acc'):
    count = tf.equal(tf.argmax(env.y, axis=1), tf.argmax(env.ybar, axis=1))
    env.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')

with tf.variable_scope('loss'):
    xent = tf.nn.softmax_cross_entropy_with_logits(labels=env.y,
                                                       logits=logits)
    env.loss = tf.reduce_mean(xent, name='loss')

with tf.variable_scope('train_op'):
    optimizer = tf.train.AdamOptimizer()
    vs = tf.global_variables()
    env.train_op = optimizer.minimize(env.loss, var_list=vs)

print('\nInitializing graph')

env.sess = tf.InteractiveSession()
env.sess.run(tf.global_variables_initializer())
env.sess.run(tf.local_variables_initializer())


def evaluate(env, X_data, y_data, batch_size=100):
    """
    Evaluate TF model by running env.loss and env.acc.
    """
    print('\nEvaluating')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    loss, acc = 0, 0

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        cnt = end - start
        batch_loss, batch_acc = env.sess.run(
            [env.loss, env.acc],
            feed_dict={env.x: X_data[start:end],
                       env.y: y_data[start:end]})
        loss += batch_loss * cnt
        acc += batch_acc * cnt
    loss /= n_sample
    acc /= n_sample

    print(' loss: {0:.4f} acc: {1:.4f}'.format(loss, acc))
    return loss, acc


def train(env, X_data, y_data, X_valid=None, y_valid=None, epochs=5,
          load=False, shuffle=True, batch_size=128, name='model'):
    if load:
        if not hasattr(env, 'saver'):
            return print('\nError: cannot find saver op')
        print('\nLoading saved model')
        env.saver.restore(env.sess, 'checkpoints/{}/{}'.format(name, name))

    print('\nTrain model')
    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    for epoch in range(epochs):
        print('\nEpoch {0}/{1}'.format(epoch + 1, epochs))

        if shuffle:
            print('\nShuffling data')
            ind = np.arange(n_sample)
            np.random.shuffle(ind)
            X_data = X_data[ind]
            y_data = y_data[ind]

        for batch in range(n_batch):
            print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
            start = batch * batch_size
            end = min(n_sample, start + batch_size)
            env.sess.run(env.train_op, feed_dict={env.x: X_data[start:end],
                                                  env.y: y_data[start:end]})
        if X_valid is not None:
            evaluate(env, X_valid, y_valid)

    if hasattr(env, 'saver'):
        print('\n Saving model')
        env.saver.save(env.sess, 'checkpoints/{}/{}'.format(name, name))


def predict(env, X_data, batch_size=128):
    print('\nPredicting')
    n_classes = env.ybar.get_shape().as_list()[1]
    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    yval = np.empty((n_sample, n_classes))

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        y_batch = env.sess.run(env.ybar, feed_dict={env.x: X_data[start:end]})
        yval[start:end] = y_batch
    print()
    return yval

print('\nTraining')
model_ref = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope='model4')  # 获取指定scope的tensor
env.saver = tf.train.Saver(model_ref)
train(env, X_train, y_train, X_valid=X_valid, y_valid=y_valid, load=False, epochs=5, name='model4')
print('\nEvaluating on clean data')
X_test = np.load("adv_images.npy") / 255.0
y_test = np.load("y_test.npy")
y_test = to_categorical(y_test)
evaluate(env, X_test, y_test)