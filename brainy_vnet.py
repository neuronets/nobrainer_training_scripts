#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 20:29:35 2021

@author: aakanksha
"""

import nobrainer
import tensorflow as tf
from nobrainer.models.bayesian_vnet import bayesian_vnet
import tensorflow.keras.backend as K

root_path = '/nobackup/users/abizeul/kwyk/tfrecords/'
train_pattern = root_path+'data-train_shard-*.tfrec'
eval_pattern = root_path + 'data-evaluate_shard-*.tfrec'

n_classes = 1
batch_size = 8
volume_shape = (256, 256, 256)
block_shape = (32, 32, 32)
n_epochs = None
augment = True
shuffle_buffer_size = 2000
num_parallel_calls = 8

dataset_train = nobrainer.dataset.get_dataset(
    file_pattern=train_pattern,
    n_classes=n_classes,
    batch_size=batch_size,
    volume_shape=volume_shape,
    block_shape=block_shape,
    n_epochs=n_epochs,
    augment=augment,
    shuffle_buffer_size=shuffle_buffer_size,
    num_parallel_calls=num_parallel_calls,
)

dataset_evaluate = nobrainer.dataset.get_dataset(
    file_pattern=eval_pattern,
    n_classes=n_classes,
    batch_size=batch_size,
    volume_shape=volume_shape,
    block_shape=block_shape,
    n_epochs=1,
    augment=False,
    shuffle_buffer_size=None,
    num_parallel_calls=1,
)
print('Data Loaded and Initialized.......')
strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))
model_path = '/nobackup/users/aakrana/nobrainer_2/nobrainer/models/brainy/vnet_brainy_weights.{epoch:03d}-{val_loss:.4f}.h5'
with strategy.scope():
    model = bayesian_vnet(input_shape = block_shape+(1,),kernel_size=3, activation="relu",padding="SAME")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-03)
    model.compile(optimizer=optimizer,loss=nobrainer.losses.jaccard,metrics=[nobrainer.metrics.dice, nobrainer.metrics.jaccard],)

print('Model Loaded and Initialized.......')

steps_per_epoch = nobrainer.dataset.get_steps_per_epoch(
    n_volumes=10000,
    volume_shape=(128,128,128),
    block_shape=block_shape,
    batch_size=batch_size)

steps_per_epoch

validation_steps = nobrainer.dataset.get_steps_per_epoch(
    n_volumes=500,
    volume_shape=(128,128,128),
    block_shape=block_shape,
    batch_size=batch_size)

validation_steps
callbacks = [tf.keras.callbacks.ModelCheckpoint(model_path)]
print('Model Training.......')

for e in range(1, 20):
    model.fit(
        dataset_train,
        steps_per_epoch=steps_per_epoch,
        validation_data=dataset_evaluate,
        validation_steps=validation_steps,
        epochs=e+1,
        initial_epoch=e,
        callbacks=callbacks)
model.save_weights('brainy_nokld.hdf5')

with strategy.scope():
    model = bayesian_vnet(input_shape = block_shape+(1,),kernel_size=3,kld =1, activation="relu",padding="SAME")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-04)
    model.compile(optimizer=optimizer,loss=nobrainer.losses.jaccard,metrics=[nobrainer.metrics.dice, nobrainer.metrics.jaccard],)
K.set_value(model.optimizer.lr, 1e-04)
for e in range(20, 40):
    model.fit(
        dataset_train,
        steps_per_epoch=steps_per_epoch,
        validation_data=dataset_evaluate,
        validation_steps=validation_steps,
        epochs=e+1,
        initial_epoch=e,
        callbacks=callbacks)

model.save_weights('brainy_kld.hdf5')
