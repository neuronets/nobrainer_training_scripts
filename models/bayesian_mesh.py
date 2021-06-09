"""
Created on Thu Feb 20 10:08:29 2021

@author: aakanksha
"""

"""Implementations of Bayesian neural networks."""

import tensorflow as tf
import tensorflow_probability as tfp

from nobrainer.layers.dropout import BernoulliDropout
from nobrainer.layers.dropout import ConcreteDropout
from nobrainer.models.bayesian_utils import normal_prior, prior_fn_for_bayesian,divergence_fn_bayesian,default_mean_field_normal_fn

tfk = tf.keras
tfkl = tfk.layers
tfpl = tfp.layers
#weightnorm = tfp.layers.weight_norm.WeightNorm # This does not work with TFP ConvVariational layers, needs to be updated with TFP update 
tfd = tfp.distributions
def variational_meshnet(
    n_classes,
    input_shape,
    receptive_field=67,
    filters=71,
    scale_factor = 3000,
    is_monte_carlo=False,
    dropout=None,
    activation=tf.nn.relu,
    batch_size=None,
    name="variational_meshnet",
):
         '''
        Steps for full training:
        1. Set priors, posteriors and divergence functions
        kl_divergence_function = divergence_fn_bayesian()
        kernel_posterior_fn = default_mean_field_normal_fn(
            loc_initializer = tf.keras.initializers.he_normal(),
            loc_regularizer=tf.keras.regularizers.l2(), #None
            untransformed_scale_regularizer=tf.keras.regularizers.l2(),
            loc_constraint= tf.keras.constraints.UnitNorm(axis = [0, 1, 2,3]), #None,
            untransformed_scale_constraint=None) #None)
        2. Put dropout regularizers in tf.compat.v1.variable_scope

         '''

    if receptive_field not in {37, 67, 129}:
        raise ValueError("unknown receptive field. Legal values are 37, 67, and 129.")
    
    def one_layer(x, layer_num, scale_factor = 3000,dilation_rate=(1, 1, 1)):
        kl_divergence_function = None#(lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                            #tf.cast(500, dtype=tf.float32))
        x = tfpl.Convolution3DFlipout(filters,
            kernel_size=3, padding="same",dilation_rate=dilation_rate,
            kernel_prior_fn=prior_fn,
            kernel_divergence_fn=kl_divergence_function, 
	    kernel_posterior_fn = kernel_posterior_fn,
            name="layer{}/vwnconv3d".format(layer_num),)(x)
        if dropout is None:
            pass
        elif dropout == "bernoulli":
            x = BernoulliDropout(
                rate=0.5,
                is_monte_carlo=is_monte_carlo,
                scale_during_training=False,
                name="layer{}/bernoulli_dropout".format(layer_num),
            )(x)
        elif dropout == "concrete":
            x = ConcreteDropout(
                is_monte_carlo=is_monte_carlo,
                temperature=0.02,
                use_expectation=is_monte_carlo,
                name="layer{}/concrete_dropout".format(layer_num),
            )(x)
        else:
            raise ValueError("unknown dropout layer, {}".format(dropout))
        x = tfkl.Activation(activation, name="layer{}/activation".format(layer_num))(x) # This activation makes no sense if Dropout is NONE! 
        return x

    inputs = tfkl.Input(shape=input_shape, batch_size=batch_size, name="inputs")
    prior_fn = normal_prior(prior_std = 1.0)
    
    if receptive_field == 37:
        x = one_layer(inputs, 1)
        x = one_layer(x, 2)
        x = one_layer(x, 3)
        x = one_layer(x, 4, dilation_rate=(2, 2, 2))
        x = one_layer(x, 5, dilation_rate=(4, 4, 4))
        x = one_layer(x, 6, dilation_rate=(8, 8, 8))
        x = one_layer(x, 7)
    elif receptive_field == 67:
        x = one_layer(inputs, 1)
        x = one_layer(x, 2)
        x = one_layer(x, 3, dilation_rate=(2, 2, 2))
        x = one_layer(x, 4, dilation_rate=(4, 4, 4))
        x = one_layer(x, 5, dilation_rate=(8, 8, 8))
        x = one_layer(x, 6, dilation_rate=(16, 16, 16))
        x = one_layer(x, 7)
    elif receptive_field == 129:
        x = one_layer(inputs, 1)
        x = one_layer(x, 2, dilation_rate=(2, 2, 2))
        x = one_layer(x, 3, dilation_rate=(4, 4, 4))
        x = one_layer(x, 4, dilation_rate=(8, 8, 8))
        x = one_layer(x, 5, dilation_rate=(16, 16, 16))
        x = one_layer(x, 6, dilation_rate=(32, 32, 32))
        x = one_layer(x, 7)

    x = tfpl.Convolution3DFlipout(
        filters=n_classes,
        kernel_size=1,
	kernel_divergence_fn = None,
	kernel_prior_fn=prior_fn,
        padding="same",
        name="classification/vwnconv3d",
    )(x)

    final_activation = "sigmoid" if n_classes == 1 else "softmax"
    x = tfkl.Activation(final_activation, name="classification/activation")(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name=name)
