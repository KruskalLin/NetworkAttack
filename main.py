import warnings
from distutils.version import LooseVersion
from config import compute_ssim, get_models, models_dict, whash, get_labels, ImageHash
import tensorflow as tf
import numpy as np
import matplotlib

matplotlib.use('Agg')
from PIL import Image
import pickle
import time
import os
from perlin import create_perlin_noise

slim = tf.contrib.slim
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
max_epsilon = 20
num_iter = 100
momentum = 1
model_index = [8, 14, 13, 3, 2, 6, 10, 9]
rand_index = [0, 3]
eps = max_epsilon / num_iter
configs = {
    'batch_size': 64,
    'epoch': 5
}
model_ind, models = get_models(model_index)


def op_with_scalar_cast(a, b, f):
    try:
        return f(a, b)
    except (TypeError, ValueError):
        pass


def mul(a, b):
    def multiply(a, b):
        return a * b

    return op_with_scalar_cast(a, b, multiply)


def optimize_linear(grad, eps, ord=np.inf):
    red_ind = list(range(1, len(grad.get_shape())))
    avoid_zero_div = 1e-12
    if ord == np.inf:
        optimal_perturbation = tf.sign(grad)
        optimal_perturbation = tf.stop_gradient(optimal_perturbation)
    elif ord == 1:
        abs_grad = tf.abs(grad)
        sign = tf.sign(grad)
        max_abs_grad = tf.reduce_max(abs_grad, red_ind, keepdims=True)
        tied_for_max = tf.to_float(tf.equal(abs_grad, max_abs_grad))
        num_ties = tf.reduce_sum(tied_for_max, red_ind, keepdims=True)
        optimal_perturbation = sign * tied_for_max / num_ties
    elif ord == 2:
        square = tf.maximum(avoid_zero_div,
                            reduce_sum(tf.square(grad),
                                       reduction_indices=red_ind,
                                       keepdims=True))
        optimal_perturbation = grad / tf.sqrt(square)
    else:
        raise NotImplementedError("Only L-inf, L1 and L2 norms are "
                                  "currently implemented.")

    # Scale perturbation to be the solution for the norm=eps rather than
    # norm=1 problem
    scaled_perturbation = mul(eps, optimal_perturbation)
    return scaled_perturbation


def reduce_function(op_func, input_tensor, axis=None, keepdims=None,
                    name=None, reduction_indices=None):
    if LooseVersion(tf.__version__) < LooseVersion('1.8.0'):
        warning = "Running on tensorflow version " + \
                  LooseVersion(tf.__version__).vstring + \
                  ". Support for this version in CleverHans is deprecated " + \
                  "and may be removed on or after 2019-01-26"
        warnings.warn(warning)
        out = op_func(input_tensor, axis=axis,
                      keep_dims=keepdims, name=name,
                      reduction_indices=reduction_indices)
    else:
        out = op_func(input_tensor, axis=axis,
                      keepdims=keepdims, name=name,
                      reduction_indices=reduction_indices)
    return out


def reduce_sum(input_tensor, axis=None, keepdims=None,
               name=None, reduction_indices=None):
    return reduce_function(tf.reduce_sum, input_tensor, axis=axis,
                           keepdims=keepdims, name=name,
                           reduction_indices=reduction_indices)


def softmax_cross_entropy_with_logits(sentinel=None,
                                      labels=None,
                                      logits=None,
                                      dim=-1):
    if sentinel is not None:
        name = "softmax_cross_entropy_with_logits"
        raise ValueError("Only call `%s` with "
                         "named arguments (labels=..., logits=..., ...)"
                         % name)
    if labels is None or logits is None:
        raise ValueError("Both labels and logits must be provided.")

    try:
        labels = tf.stop_gradient(labels)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=labels, logits=logits, dim=dim)
    except AttributeError:
        warning = "Running on tensorflow version " + \
                  LooseVersion(tf.__version__).vstring + \
                  ". Support for this version in CleverHans is deprecated " + \
                  "and may be removed on or after 2019-01-26"
        warnings.warn(warning)
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, dim=dim)

    return loss


def minibatches(inputs_data, inputs_labels, batch_size=None, shuffle=False):
    assert len(inputs_data) == len(inputs_labels)
    if shuffle:
        indices = np.arange(len(inputs_data))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs_data) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs_data[excerpt], inputs_labels[excerpt]


def precalc_jitter_mask(width=3):
    # Prepare a jitter mask with XOR (alternating).
    jitter_width = width
    jitter_mask = np.empty((configs['batch_size'], 28, 28, 1), dtype=np.bool)
    for i in range(28):
        for j in range(28):
            jitter_mask[:, i, j, :] = (i % jitter_width == 0) ^ (j % jitter_width == 0)
    return tf.convert_to_tensor(jitter_mask, dtype=tf.bool)


def generate_jitter_sample(X_orig, X_aex, fade_eps=0.1, width=3):
    jitter_mask = precalc_jitter_mask(width=width)
    jitter_mask = tf.cast(jitter_mask, dtype=tf.float32)
    jitter_diff = (X_aex - X_orig) * jitter_mask
    X_candidate = X_aex - fade_eps * jitter_diff
    return X_candidate


def refine_advimages(ori, adv, pre_labels, target_lists):
    for i in range(len(target_lists)):
        if pre_labels[i] == target_lists[i]:
            adv[i, :, :, :] = ori[i, :, :, :]
    return adv


def stop(x, y, i, x_max, x_min, x_ori):
    return tf.less(i, num_iter)


def target_graph(x, y, i, x_max, x_min, x_ori):
    # x = x + tf.random_uniform(tf.shape(x), minval=-1e-2, maxval=1e-2) * \
    #     create_perlin_noise(seed=None, color=False, batch_size=configs['batch_size'], image_size=28, normalize=True,
    #                         precalc_fade=None)
    # x = generate_jitter_sample(x_ori, x, fade_eps=0, width=np.random.randint(1, 3))
    one_hot = tf.one_hot(y, 10)
    logits = []
    noises = []
    for j in rand_index:
        logit, output = models[j](x)
        logits.append(logit)
    one_hot = one_hot / reduce_sum(one_hot, 1, keepdims=True)
    for logit in logits:
        loss = softmax_cross_entropy_with_logits(labels=one_hot, logits=logit)
        grad, = tf.gradients(loss, x)
        optimal_perturbation = optimize_linear(grad, eps)
        noises.append(optimal_perturbation)
    x = x + np.sum(noises, axis=0)
    x = tf.clip_by_value(x, x_min, x_max)
    i = tf.add(i, 1)
    return x, y, i, x_max, x_min, x_ori


def cw(x, y=None, eps=2.5, ord_='inf', T=512,
       optimizer=tf.train.AdamOptimizer(learning_rate=0.1), alpha=0.9,
       min_prob=0., clip=(0.0, 1.0)):
    xshape = x.get_shape().as_list()
    noise = tf.get_variable('noise', xshape, tf.float32,
                            initializer=tf.initializers.zeros)
    x = x + tf.random_uniform(tf.shape(x), minval=-1e-2, maxval=1e-2) * \
        create_perlin_noise(seed=None, color=False, batch_size=configs['batch_size'], image_size=28, normalize=True,
                            precalc_fade=None)
    # scale input to (0, 1)
    x_scaled = (x - clip[0]) / (clip[1] - clip[0])

    # change to sigmoid-space, clip to avoid overflow.
    z = tf.clip_by_value(x_scaled, 1e-8, 1 - 1e-8)
    xinv = tf.log(z / (1 - z))

    # add noise in sigmoid-space and map back to input domain
    xadv = tf.sigmoid(xinv + T * noise)
    xadv = xadv * (clip[1] - clip[0]) + clip[0]

    logits = []
    ybars = []
    for i in rand_index:
        logit, ybar = models[i](xadv)
        logits.append(logit)
        ybars.append(ybar)
    ybar = tf.add_n(ybars)
    logits = tf.add_n(logits)
    ydim = ybar.get_shape().as_list()[1]

    if y is not None:
        y_temp = tf.cond(tf.equal(tf.rank(y), 0),
                         lambda: tf.fill([xshape[0]], y),
                         lambda: tf.identity(y))
    else:
        # we set target to the least-likely label
        y_temp = tf.argmin(ybar, axis=1, output_type=tf.int32)

    mask = tf.one_hot(y_temp, ydim, on_value=0.0, off_value=float('inf'))
    yt = tf.reduce_max(logits - mask, axis=1)
    yo = tf.reduce_max(logits, axis=1)

    # encourage to classify to a wrong category
    loss0 = tf.nn.relu(yo - yt + min_prob)

    axis = list(range(1, len(xshape)))
    ord_ = float(ord_)

    # make sure the adversarial images are visually close
    if 2 == ord_:
        # CW-L2 Original paper uses the reduce_sum version.  These two
        # implementation does not differ much.

        # loss1 = tf.reduce_sum(tf.square(xadv-x), axis=axis)
        loss1 = tf.reduce_mean(tf.square(xadv - x))
    else:
        # CW-Linf
        tau0 = tf.fill([xshape[0]] + [1] * len(axis), clip[1])
        tau = tf.get_variable('cw8-noise-upperbound', dtype=tf.float32,
                              initializer=tau0, trainable=False)
        diff = xadv - x - tau

        # if all values are smaller than the upper bound value tau, we reduce
        # this value via tau*0.9 to make sure L-inf does not get stuck.
        tau = alpha * tf.to_float(tf.reduce_all(diff < 0, axis=axis))
        loss1 = tf.nn.relu(tf.reduce_sum(diff, axis=axis))

    loss = eps * loss0 + loss1
    train_op = optimizer.minimize(loss, var_list=[noise])

    # We may need to update tau after each iteration.  Refer to the CW-Linf
    # section in the original paper.
    if 2 != ord_:
        train_op = tf.group(train_op, tau)

    return train_op, xadv, noise


# def save_images(images, filenames, output_dir):
#     for i, filename in enumerate(filenames):
#         with open(os.path.join(output_dir, filename), 'wb') as f:
#             image = (((images[i] + 1.0) * 0.5) * 255.0).astype(np.uint8)
#             # resize back to [299, 299]
#             image = imresize(image, [299, 299])
#             Image.fromarray(image).save(f, format='PNG')


def eval(adv_imgs, labels, x_inputs, total_count, total_score):
    adv_preds = []
    preds = []
    for model in models:
        logit, output = model(x_inputs)
        preds.append(tf.argmax(output, 1))

    for model in models:
        logit, output = model(adv_imgs)
        adv_preds.append(tf.argmax(output, 1))

    for i in range(adv_imgs.shape[0]):
        def f1(total_count):
            return total_count

        def f2(total_count):
            total_count = tf.add(total_count, 1)
            return total_count

        def f3(total_count, x_ori, x_adv):
            total_count = tf.add(total_count, tf.image.ssim(x_ori * 255, x_adv * 255, max_val=255))
            return total_count

        for j in range(len(adv_preds)):
            total_count = tf.cond(tf.equal(adv_preds[j][i], labels[i]), lambda: f2(total_count),
                                  lambda: f1(total_count))
            total_score = tf.cond(tf.logical_and(tf.equal(preds[j][i], labels[i]), tf.not_equal(adv_preds[j][i], labels[i])),
                                  lambda: f3(total_score, x_inputs[i], adv_imgs[i]), lambda: f1(total_score))
    return total_count, total_score


def optimistic_restore(session, save_file, graph=tf.get_default_graph()):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                        if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    for var_name, saved_var_name in var_names:
        curr_var = graph.get_tensor_by_name(var_name)
        var_shape = curr_var.get_shape().as_list()
        if var_shape == saved_shapes[saved_var_name]:
            restore_vars.append(curr_var)
    opt_saver = tf.train.Saver(restore_vars)
    opt_saver.restore(session, save_file)


def aiTest(images, shape):
    batch_shape = [configs['batch_size'], 28, 28, 1]
    X_test = images / 255.0
    y_test = get_labels(images)
    np.save('y_test', y_test)
    if shape[0] % configs['batch_size'] != 0:
        diff = configs['batch_size'] - shape[0] % configs['batch_size']
        X_test = np.concatenate([X_test, np.zeros((diff, 28, 28, 1), dtype=np.uint8)], axis=0)
        y_test = np.append(y_test, np.ones((diff, 1), dtype=np.int64) * 3)
    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        x_ori = tf.placeholder(tf.float32, shape=batch_shape)
        x_max = tf.clip_by_value(x_input + eps, 0.0, 1.0)
        x_min = tf.clip_by_value(x_input - eps, 0.0, 1.0)

        y = tf.placeholder(tf.int64, shape=[configs['batch_size']])
        i = tf.constant(0)
        # model_ind, models = get_models(model_index)
        adv_x, _, _, _, _, _ = tf.while_loop(stop, target_graph,
                                             [x_input, y, i, x_max, x_min, x_ori])
        # optimizer = tf.train.AdamOptimizer(learning_rate=0.2)
        # adv_train_op, adv_x, noise = cw(x_input, optimizer=optimizer)
        # eval
        adv_input = tf.placeholder(tf.float32, shape=batch_shape)
        labels = tf.placeholder(tf.int64, shape=[configs['batch_size']])
        batch_count = tf.constant(0.0)
        batch_score = tf.constant(0.0)
        count = eval(adv_input, labels, x_input, batch_count, batch_score)
        total_count = 0
        total_score = 0.0
        savers = [(ind, tf.train.Saver(slim.get_model_variables(scope=ind))) for ind in model_ind]
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            for (ind, saver) in savers:
                saver.restore(sess, 'checkpoints/' + ind + '/' + ind)
            adv_images = []
            start_time = time.time()
            for x, label in minibatches(X_test, y_test, configs['batch_size'], shuffle=False):
                x_adv = sess.run(adv_x, feed_dict={x_input: x, y: label, x_ori: x})
                adv_images.append(x_adv)
                sub_count, sub_score = sess.run(count, feed_dict={adv_input: x_adv, labels: label, x_input: x})
                total_count = total_count + sub_count
                total_score = total_score + sub_score
            adv_images = np.concatenate(adv_images, axis=0)
            adv_images = adv_images[0:shape[0]] * 255
        elapsed_time = time.time() - start_time
    print('Finish! Time:{}'.format(elapsed_time))
    print('total score: {}'.format(total_score))
    return adv_images


if __name__ == '__main__':
    images = np.load("test_image_cases.npy")
    # img_size = 28
    # img_chan = 1
    # mnist = tf.keras.datasets.fashion_mnist
    # (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # X_test = np.reshape(X_test, [-1, img_size, img_size, img_chan])
    # y_test = y_test.astype(np.int64)
    # indices = np.arange(len(X_test))
    # np.random.shuffle(indices)
    # images = X_test[indices[0:10000]]
    adv_images = aiTest(images, (1000, 28, 28, 1))
    print(adv_images.shape)
    adv_images = np.trunc(adv_images)
    np.save('adv_images', adv_images)

    ssim = []
    for i in range(len(images)):
        ssim.append(compute_ssim(np.squeeze(images[i], axis=2), np.squeeze(adv_images[i], axis=2)))
    print(np.average(ssim))