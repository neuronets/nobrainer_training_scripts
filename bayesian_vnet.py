from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate
from tensorflow.keras.models import Model
import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow.python.keras import backend as K
from nobrainer.layers.groupnorm import GroupNormalization
from nobrainer.models.bayesian_utils import normal_prior, prior_fn_for_bayesian,divergence_fn_bayesian
tfd = tfp.distributions

def down_stage(inputs, filters, kernel_size=3,
             activation="relu", padding="SAME"):
    conv = Conv3D(filters, kernel_size,
                  activation=activation, padding=padding)(inputs)
    conv = GroupNormalization()(conv)
    conv = Conv3D(filters, kernel_size,
                  activation=activation, padding=padding)(conv)
    conv = GroupNormalization()(conv)
    pool = MaxPooling3D()(conv)
    return conv, pool


def up_stage(inputs, skip, filters, prior_fn, kernel_posterior_fn, 
             kld, kernel_size=3, activation="relu", padding="SAME"):
#    if K.get_value(kld_cp) == 0:
#    kld = divergence_fn_bayesian(prior_std=1.0, examples_per_epoch=3000)
##    else:
##        kld = (lambda q, p, _: K.get_value(kld_cp)*tfd.kl_divergence(q, p)/tf.cast(3000, dtype=tf.float32))
#    kernel_posterior_fn = tfp.layers.default_mean_field_normal_fn(
#            loc_initializer = tf.keras.initializers.he_normal(),
#            loc_regularizer=tf.keras.regularizers.l2(), #None
#            untransformed_scale_regularizer=tf.keras.regularizers.l2(),
#            loc_constraint = tf.keras.constraints.UnitNorm(axis = [0, 1, 2,3]),#None,
#            untransformed_scale_constraint=None) #None)
    up = UpSampling3D()(inputs)
    up = tfp.layers.Convolution3DFlipout(filters, 2,
                                         activation=activation,
                                         padding=padding,
                                         kernel_divergence_fn = kld,
                                         kernel_posterior_fn = kernel_posterior_fn,
                                         kernel_prior_fn=prior_fn)(up)
    up = GroupNormalization()(up)

    merge = concatenate([skip, up])
    merge = GroupNormalization()(merge)

    conv = tfp.layers.Convolution3DFlipout(filters, kernel_size,
                                           activation=activation,
                                           padding=padding,
                                           kernel_divergence_fn=kld,
                                           kernel_posterior_fn = kernel_posterior_fn,
                                           kernel_prior_fn=prior_fn)(merge)
    conv = GroupNormalization()(conv)
    conv = tfp.layers.Convolution3DFlipout(filters, kernel_size,
                                           activation=activation,
                                           padding=padding,
                                           kernel_divergence_fn=kld,
                                           kernel_posterior_fn = kernel_posterior_fn,
                                           kernel_prior_fn=prior_fn)(conv)
    conv = GroupNormalization()(conv)

    return conv


def end_stage(inputs, prior_fn, kernel_posterior_fn, kld, kernel_size=3,
              activation="relu", padding="SAME"):
    #if K.get_value(kld_cp) == 0:
    #kld = divergence_fn_bayesian(prior_std=1.0, examples_per_epoch = 3000)
#    else:
#        kld = (lambda q, p, _: K.get_value(kld_cp)*tfd.kl_divergence(q, p)/tf.cast(3000, dtype=tf.float32))
#    kernel_posterior_fn = tfp.layers.default_mean_field_normal_fn(
#            loc_initializer = tf.keras.initializers.he_normal(),
#            loc_regularizer=tf.keras.regularizers.l2(), #None
#            untransformed_scale_regularizer=tf.keras.regularizers.l2(),
#            loc_constraint= tf.keras.constraints.UnitNorm(axis = [0, 1, 2,3]), #None,
#            untransformed_scale_constraint=None) #None)
    conv = tfp.layers.Convolution3DFlipout(1, kernel_size,
                                           activation=activation,
                                           padding="SAME",
                                           kernel_divergence_fn = kld,
                                           kernel_posterior_fn = kernel_posterior_fn,
                                           kernel_prior_fn=prior_fn)(inputs)
    conv = tfp.layers.Convolution3DFlipout(1, 1, activation="sigmoid",
                                           kernel_divergence_fn = kld,
                                           kernel_posterior_fn = kernel_posterior_fn,
                                           kernel_prior_fn=prior_fn)(conv)
    return conv


def bayesian_vnet(input_shape, kernel_size=3, prior_fn = prior_fn_for_bayesian(),
                  kernel_posterior_fn = tfp.layers.default_mean_field_normal_fn(),
                  kld = None ,
                  activation="relu", padding="SAME"):
    if kld:
        kld = divergence_fn_bayesian(prior_std=.1, examples_per_epoch = 3000)
    inputs = Input(input_shape)

    conv1, pool1 = down_stage(inputs, 16,
                              kernel_size=kernel_size,
                              activation=activation,
                              padding=padding)
    conv2, pool2 = down_stage(pool1, 32,
                              kernel_size=kernel_size,
                              activation=activation,
                              padding=padding)
    conv3, pool3 = down_stage(pool2, 64,
                              kernel_size=kernel_size,
                              activation=activation,
                              padding=padding)
    conv4, _ = down_stage(pool3, 128,
                          kernel_size=kernel_size,
                          activation=activation,
                          padding=padding)

    conv5 = up_stage(conv4, conv3, 64, prior_fn, 
                     kernel_posterior_fn, kld,
                     kernel_size=kernel_size,
                     activation=activation,
                     padding=padding)
    conv6 = up_stage(conv5, conv2, 32, prior_fn, 
                     kernel_posterior_fn, kld,
                     kernel_size=kernel_size,
                     activation=activation,
                     padding=padding)
    conv7 = up_stage(conv6, conv1, 16, prior_fn, 
                     kernel_posterior_fn, kld,
                     kernel_size=kernel_size,
                     activation=activation,
                     padding=padding)

    conv8 = end_stage(conv7, prior_fn, kernel_posterior_fn, kld,
                      kernel_size= kernel_size,
                      activation=activation,
                      padding=padding)

    return Model(inputs=inputs, outputs=conv8)


#from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate
#from tensorflow.keras.models import Model
#import tensorflow_probability as tfp
#import tensorflow as tf
#from nobrainer.layers.groupnorm import GroupNormalization
#from nobrainer.models.bayesian_utils import normal_prior
#tfd = tfp.distributions
#
#def prior_fn_for_bayesian(init_scale_mean=-1, init_scale_std=0.1):
#    def prior_fn(dtype, shape, name, _, add_variable_fn):
#        untransformed_scale = add_variable_fn(name=name + '_untransformed_scale',
#                shape=(1,), initializer=tf.compat.v1.initializers.random_normal(
#                mean=init_scale_mean, stddev=init_scale_std), dtype=dtype,
#                trainable = False)
#        loc = add_variable_fn(name=name + '_loc',initializer=tf.keras.initializers.Zeros(),
#        shape=shape,dtype=dtype, trainable=True)
#        scale = 1e-6 + tf.nn.softplus(untransformed_scale)
#        dist = tfd.Normal(loc=loc, scale=scale)
#        batch_ndims = tf.size(input=dist.batch_shape_tensor())
#        return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)
#    return prior_fn
##
#def divergence_fn_bayesian(prior_std = 0.1, examples_per_epoch=2000):
#    def divergence_fn(q, p, _):
#        log_probs = tfd.LogNormal(0., prior_std).log_prob(p.stddev())
#        out = tfd.kl_divergence(q, p) - tf.reduce_sum(log_probs)
#        return out / examples_per_epoch
#    return divergence_fn
#
#def down_stage(inputs, filters, kernel_size=3,
#             activation="relu", padding="SAME"):
#    conv = Conv3D(filters, kernel_size,
#                  activation=activation, padding=padding)(inputs)
#    conv = GroupNormalization()(conv)
#    conv = Conv3D(filters, kernel_size,
#                  activation=activation, padding=padding)(conv)
#    conv = GroupNormalization()(conv)
#    pool = MaxPooling3D()(conv)
#    return conv, pool
#
#
#def up_stage(inputs, skip, filters, prior_fn, kernel_size=3,
#               activation="relu", padding="SAME"):
#    up = UpSampling3D()(inputs)
#    kld= None
#    #kld = divergence_fn_bayesian()
##    kld= (lambda q, p, _: tfd.kl_divergence(q, p)/tf.cast(3000, dtype=tf.float32))
#    kernel_posterior_fn = tfp.layers.default_mean_field_normal_fn(
#            loc_initializer = tf.keras.initializers.he_normal(),
#            untransformed_scale_initializer= tf.constant_initializer(0.0001),
#            loc_regularizer= tf.keras.regularizers.l2(), #None tf.nn.scale_regularization_loss(tf.keras.regularizers.l2())
##            untransformed_scale_regularizer= tf.keras.regularizers.l2(),
##            loc_constraint = tf.keras.constraints.UnitNorm(axis = [ 0,1, 2,3]),#None,
#            untransformed_scale_constraint=None) #None)
#    
#    up = tfp.layers.Convolution3DFlipout(filters, 2,
#                                         activation=activation,
#                                         padding=padding,
#                                         kernel_divergence_fn=kld,
#  #                                       kernel_posterior_fn = kernel_posterior_fn,
#                                         kernel_prior_fn=prior_fn)(up)
#    up = GroupNormalization()(up)
#
#    merge = concatenate([skip, up])
#    merge = GroupNormalization()(merge)
#
#    conv = tfp.layers.Convolution3DFlipout(filters, kernel_size,
#                                           activation=activation,
#                                           padding=padding,
#                                           kernel_divergence_fn = kld,
#                                         kernel_posterior_fn = kernel_posterior_fn,
#                                           kernel_prior_fn=prior_fn)(merge)
#    conv = GroupNormalization()(conv)
#    conv = tfp.layers.Convolution3DFlipout(filters, kernel_size,
#                                           activation=activation,
#                                           padding=padding,
#                                           kernel_divergence_fn=kld,
##                                           kernel_posterior_fn =  kernel_posterior_fn,
#                                           kernel_prior_fn=prior_fn)(conv)
#    conv = GroupNormalization()(conv)
#
#    return conv
#
#
#def end_stage(inputs, prior_fn, kernel_size=3,
#              activation="relu", padding="SAME"):
#    kld= None
##    kld = (lambda q, p, _: tfd.kl_divergence(q, p)/tf.cast(3000, dtype=tf.float32))
#    #kld = divergence_fn_bayesian()
#    kernel_posterior_fn = tfp.layers.default_mean_field_normal_fn(
#            loc_initializer = tf.keras.initializers.he_normal(),
#            untransformed_scale_initializer= tf.constant_initializer(0.0001),
#            loc_regularizer=  tf.keras.regularizers.l2(), #None
#            untransformed_scale_regularizer= tf.keras.regularizers.l2(),
#            loc_constraint = tf.keras.constraints.UnitNorm(axis = [0,1, 2,3]),#None,
#            untransformed_scale_constraint=None) #None)
#
#    conv = tfp.layers.Convolution3DFlipout(1, kernel_size,
#                                           activation=activation,
#                                           padding="SAME",
#                                           kernel_divergence_fn = kld,
##                                           kernel_posterior_fn = kernel_posterior_fn,
#                                           kernel_prior_fn=prior_fn)(inputs)
#    conv = tfp.layers.Convolution3DFlipout(1, 1, activation="sigmoid",
#                                           kernel_divergence_fn = kld,
##                                           kernel_posterior_fn = kernel_posterior_fn,
#                                           kernel_prior_fn=prior_fn)(conv)
#
#    return conv
#
#
#def bayesian_vnet(input_shape=(280, 280, 280, 1), kernel_size=3,
#                  activation="relu", padding="SAME"):
#    #prior_fn = prior_fn_for_bayesian()
#    prior_fn = normal_prior(prior_std=.1)#
#
#    inputs = Input(input_shape)
#
#    conv1, pool1 = down_stage(inputs, 16,
#                              kernel_size=kernel_size,
#                              activation=activation,
#                              padding=padding)
#    conv2, pool2 = down_stage(pool1, 32,
#                              kernel_size=kernel_size,
#                              activation=activation,
#                              padding=padding)
#    conv3, pool3 = down_stage(pool2, 64,
#                              kernel_size=kernel_size,
#                              activation=activation,
#                              padding=padding)
#    conv4, _ = down_stage(pool3, 128,
#                          kernel_size=kernel_size,
#                          activation=activation,
#                          padding=padding)
#
#    conv5 = up_stage(conv4, conv3, 64, prior_fn,
#                     kernel_size=kernel_size,
#                     activation=activation,
#                     padding=padding)
#    conv6 = up_stage(conv5, conv2, 32, prior_fn,
#                     kernel_size=kernel_size,
#                     activation=activation,
#                     padding=padding)
#    conv7 = up_stage(conv6, conv1, 16, prior_fn,
#                     kernel_size=kernel_size,
#                     activation=activation,
#                     padding=padding)
#
#    conv8 = end_stage(conv7, prior_fn,
#                      kernel_size=kernel_size,
#                      activation=activation,
#                      padding=padding)
#
#    return Model(inputs=inputs, outputs=conv8)
