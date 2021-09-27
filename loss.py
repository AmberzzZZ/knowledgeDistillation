import keras.backend as K
import tensorflow as tf


def ce_loss(args):
    gt, logits = args[:2]
    probs = K.softmax(logits)
    ce_loss = K.categorical_crossentropy(gt, probs)   # (b,)
    return K.expand_dims(ce_loss, axis=1)   # (b,1)


def att_loss(args):
    y_s, y_t = args[:2]    # (b,cls)
    feats_s = args[2:6]    # (b,h,w,c)
    feats_t = args[6:10]

    alpha = 2
    beta = 100

    # label consistency: kl divergence
    loss_kd = kl_div(y_s, y_t)   # (b,1)

    # feature consistency: l2 distance of the pooling_norm
    loss_f = tf.zeros_like(loss_kd)
    i = 2   # stage level
    for f_s, f_t in zip(feats_s, feats_t):
        l2_distance = K.pow(pooling_norm(f_s)-pooling_norm(tf.stop_gradient(f_t)), 2)
        loss_f += K.mean(l2_distance, axis=1, keepdims=True)
        i += 1

    loss = alpha * loss_kd + beta * loss_f

    return loss


def kl_div(y_s, y_t):      # logits, (b,cls)
    temperature = 4.
    # log probs
    log_s, log_t = tf.nn.log_softmax(y_s/temperature), tf.nn.log_softmax(y_t/temperature)
    # probs
    p_s = tf.exp(log_s)
    return K.sum(p_s*(log_s-log_t), axis=1, keepdims=True)


def pooling_norm(x):
    # channel-wise pooling
    x = K.mean(K.pow(x, 2), axis=-1)    # (b,h,w)
    b = tf.shape(x)[0]
    x = K.reshape(x, (b,-1))   # (b,hw)
    # l2 norm
    x = tf.nn.l2_normalize(x, axis=1, epsilon=1e-12)    # l2-normed (b,hw), x^2 = 1
    return x







