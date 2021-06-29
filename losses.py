import tensorflow as tf
import nobrainer
import numpy as np
import tensorflow.keras.backend as K

def generalized_dice_re(y_true, y_pred, axis=(1, 2, 3)):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    if y_true.get_shape().ndims < 2 or y_pred.get_shape().ndims < 2:
        raise ValueError("y_true and y_pred must be at least rank 2.")

    epsilon = tf.keras.backend.epsilon()
    
    w = tf.math.reciprocal(tf.square(tf.reduce_sum(y_true, axis=axis)))
    w = tf.where(tf.math.is_finite(w), w, epsilon)
    num = 2 * tf.reduce_sum(w * tf.reduce_sum(y_true * y_pred, axis= axis), axis=-1)
    den = tf.reduce_sum(w * tf.reduce_sum(y_true + y_pred, axis= axis), axis=-1)
    gdice = num/den
    gdice = tf.where(tf.math.is_finite(gdice), gdice, tf.zeros_like(gdice))
    return gdice

def g_dice(y_true, y_pred, axis=(1, 2, 3)):
    gdice = 1.0 - generalized_dice(y_true=y_true, y_pred=y_pred, axis=axis)
    ce =  tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=y_true, y_pred=y_pred))
    return gdice + ce 

def dice_coef_multilabel(y_true, y_pred):
    n_classes= tf.shape(y_pred)[-1]
    dice=0
    for index in range(n_classes):
        dice -= dice_coef(y_true[:,:,:,:,index], y_pred[:,:,:,:,index])
    return dice

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1e-8) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1e-8)
