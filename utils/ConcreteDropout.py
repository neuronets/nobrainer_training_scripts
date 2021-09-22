from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

tfk = tf.keras
tfkl = tfk.layers
import tensorflow_probability as tfp
tfd= tfp.distributions

class ConcreteDropout(tf.keras.layers.Layer):
  def __init__(
        self,
        is_monte_carlo,
        filters,
        temperature=0.01,
        use_expectation= tf.constant(True,dtype=tf.bool),
        scale_factor = 1,
        seed=None,
        **kwargs):

    super(ConcreteDropout, self).__init__(
        **kwargs)
    self.is_monte_carlo = is_monte_carlo
    self.filters = filters
    self.temperature = temperature
    self.use_expectation = use_expectation,
    self.seed = seed
    self.scale_factor = scale_factor
    
    
  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    #if self.concrete:
    p_prior = tfk.initializers.Constant(0.5)
    initial_p = tfk.initializers.Constant(0.9)
    self.p_post = self.add_variable(
            name="p_l", shape=input_shape[-1:], initializer=initial_p, trainable= True)
    
    self.p_prior = self.add_variable(
            name="pr", shape=input_shape[-1:], initializer=p_prior, trainable=False)
    self.p_post = tfk.backend.clip(self.p_post, 0.05, 0.95)
    self.built = True

  def call(self, inputs):
    outputs = self._apply_concrete(inputs)
    self._apply_divergence_concrete(self.scale_factor, name='concrete_loss')
    return outputs

  def compute_output_shape(self, input_shape):
      return input_shape
      
  def get_config(self):
      config = super().get_config()
      config.update(
            {
                "is_monte_carlo": self.is_monte_carlo,
                "temperature": self.temperature,
                "use_expectation": self.use_expectation,
                "seed": self.seed,
                "scale":self.scale_factor,
            }
        )
      return config    

  @classmethod
  def from_config(cls, config):
    config = config.copy()
    return cls(**config)
  def _apply_concrete(self, outputs):
      inference = outputs
      eps = tfk.backend.epsilon()
      use_expectation = self.use_expectation

      if self.is_monte_carlo:
          noise = tf.random.uniform(tf.shape(inference),
                minval=0, maxval=1,
                seed=None,
                dtype=self.dtype)
          z = tf.nn.sigmoid((tf.math.log(self.p_post + eps)
                    - tf.math.log(1.0 - self.p_post + eps)
                    + tf.math.log(noise + eps)
                    - tf.math.log(1.0 - noise + eps)
                )/ self.temperature
            )
          return outputs * z
      else:
          return inference * self.p_post if use_expectation else inference
  def _apply_divergence_concrete(self,scale_factor, name):
      divergence_fn = (lambda pl, pr: (tf.reduce_sum(tf.add(
                tf.multiply(pl,tf.subtract(tf.math.log(
                        tf.add(pl,tfk.backend.epsilon())), tf.math.log(pr))),
                tf.multiply( tf.subtract(tfk.backend.constant(1),pl), 
                    tf.subtract(tf.math.log(
                         tf.add(tf.subtract(tfk.backend.constant(1),pl),tfk.backend.epsilon())),
                        tf.math.log(pr))))  )
                 /tf.cast(scale_factor, dtype=tf.float32)))
      divergence = tf.identity(
            divergence_fn(self.p_post, self.p_prior),
            name=name)
      self.add_loss(divergence)
      
