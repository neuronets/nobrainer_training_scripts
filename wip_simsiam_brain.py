"""
Extending the SimSiam network architecture to brain volumes
author: Dhritiman Das
"""

import tensorflow as tf

from tensorflow.keras import layers, regularizers, activations
import nobrainer
import numpy as np

def simsiam_brain(
    input_shape_1 = (256, 256, 256, 1),
    input_shape_2 = (256, 256, 256, 1),
    n_classes = 1,
    weight_decay = 0.0005,
    projection_dim = 2048,
    latent_dim = 512,
    initial_learning_rate=0.03,
):
      # zip both the augmented datasets
    #input_shape = tf.data.Dataset.zip((input_shape_1, input_shape_2))
    

    #print("augment_one: ", augment_one)
    #print("augment_two: ", augment_two)


    # define the encoder and projector: this is built on the highresnet backbone
    def encoder():
        resnet = nobrainer.models.highresnet(n_classes=n_classes, input_shape=input_shape_1,)
           
        input = tf.keras.layers.Input(shape=input_shape_1)

        resnet_out = resnet(input)
        
        x = layers.GlobalAveragePooling3D(name="backbone_pool")(resnet_out)

        x = layers.Dense(
                projection_dim, use_bias=False, kernel_regularizer=regularizers.l2(weight_decay)
            )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dense(
                projection_dim, use_bias=False, kernel_regularizer=regularizers.l2(weight_decay)
            )(x)
        output = layers.BatchNormalization()(x)

        encoder_model = tf.keras.Model(input, output, name="encoder")
        return encoder_model

        #exm_encoder = encoder()
        #exm_encoder.summary() #view encoder details

    # define predictor

    def predictor():
        model = tf.keras.Sequential(
                [
                    # Note the AutoEncoder-like structure.
                    tf.keras.layers.InputLayer((projection_dim,)),
                    tf.keras.layers.Dense(
                        latent_dim, 
                        use_bias=False,
                        kernel_regularizer=regularizers.l2(weight_decay),
                        ),
                    tf.keras.layers.ReLU(),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dense(projection_dim),
                ],
                name="predictor",
            )
        return model

    simsiam = SimSiam(encoder(), predictor())
    return simsiam

class SimSiam(tf.keras.Model):
    def __init__(self, encoder, predictor):
        super(SimSiam, self).__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, input_shape_1, input_shape_2):
        
        self.input_shape_1 = input_shape_1
        self.input_shape_2 = input_shape_2
        

        # Forward pass through the encoder and predictor.
        with tf.GradientTape() as tape:
            z1, z2 = self.encoder(self.input_shape_1), self.encoder(self.input_shape_2)
            p1, p2 = self.predictor(z1), self.predictor(z2)
            loss =  compute_loss(p1, z2) / 2 + compute_loss(p2, z1) / 2

        # Compute gradients and update the parameters.
        learnable_params = (
            self.encoder.trainable_variables + self.predictor.trainable_variables
        )
        gradients = tape.gradient(loss, learnable_params)
        self.optimizer.apply_gradients(zip(gradients, learnable_params))

        # Monitor loss.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
        
def compute_loss(p, z):    
    z = tf.stop_gradient(z)
    p = tf.math.l2_normalize(p, axis=1)
    z = tf.math.l2_normalize(z, axis=1)

    # Negative cosine similarity loss
    return -tf.reduce_mean(tf.reduce_sum((p * z), axis=1))
      

    


   # Compile model and start training.
    # EPOCHS = 1 #should be higher say >100
    # lr_decayed_fn = tf.keras.experimental.CosineDecay(
    # initial_learning_rate=0.03, decay_steps=steps
    # )

    # simsiam = SimSiam(encoder(), predictor())
    # simsiam.compile(optimizer=tf.keras.optimizers.SGD(lr_decayed_fn, momentum=0.6))
    # history = simsiam.fit(augment_data, epochs=EPOCHS, steps_per_epoch = pretrain_steps, callbacks=[early_stopping])

