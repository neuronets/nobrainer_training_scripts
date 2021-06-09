#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 13:15:33 2021

@author: aakanksha
"""

import tensorflow as tf
import nobrainer
import glob
import numpy as np
import pandas as pd
from nobrainer.models.bayesian_mesh import variational_meshnet
from tensorflow.keras.callbacks import ModelCheckpoint
from nobrainer.models.bayesian_utils import normal_prior, prior_fn_for_bayesian

def _to_blocks(x, y,block_shape):
    """Separate `x` into blocks and repeat `y` by number of blocks."""
    print(x.shape)
    x = nobrainer.volume.to_blocks(x, block_shape)
    y = nobrainer.volume.to_blocks(y, block_shape)
    return (x, y)

def get_dict(n_classes):
    print('Conversion into {} segmentation classes from freesurfer labels to 0-{}'.format(n_classes,n_classes-1))
    if n_classes == 49: 
        tmp = pd.read_csv('50-class-mapping.csv', header=0,usecols=[1,2],dtype=np.int32)
        tmp = tmp.iloc[1:,:] # removing the unknown class
        mydict = dict(tuple(zip(tmp['original'],tmp['new'])))
        return mydict
    elif n_classes == 115:
        tmp = pd.read_csv('115-class-mapping.csv', header=0,usecols=[0,1],dtype=np.int32)
        mydict = dict(tuple(zip(tmp['original'],tmp['new'])))
        del tmp
        return mydict
    else: raise(NotImplementedError)
    
def process_dataset(dset,batch_size,block_shape,n_classes,one_hot_label= False,training= True):
    # Standard score the features.
    dset = dset.map(lambda x, y: (nobrainer.volume.standardize(x), nobrainer.volume.replace(y,get_dict(n_classes))))
    
    # Separate features into blocks.
    dset = dset.map(lambda x, y:_to_blocks(x,y,block_shape))
    if one_hot_label:
        dset= dset.map(lambda x,y:(x, tf.one_hot(y,n_classes)))
    # This step is necessary because separating into blocks adds a dimension.
    dset = dset.unbatch()
    if training:
        dset = dset.shuffle(buffer_size=100)
    # Add a grayscale channel to the features.
    dset = dset.map(lambda x, y: (tf.expand_dims(x, -1), y))
    # Batch features and labels.
    dset = dset.batch(batch_size, drop_remainder=True)
    return dset

def get_dataset(pattern,volume_shape,batch,block_shape,n_classes,one_hot_label= False,training = True):

    dataset = nobrainer.dataset.tfrecord_dataset(
        file_pattern=glob.glob(pattern),
        volume_shape=volume_shape,
        shuffle=False,
        scalar_label=False,
        compressed=True)
    dataset = process_dataset(dataset,batch,block_shape,n_classes, one_hot_label= one_hot_label ,training = training)
    return dataset

root_path = '/nobackup/users/aakrana/data/kwyk/' #'/nobackup/users/abizeul/kwyk/tfrecords/'
train_pattern = root_path+'data-train_shard-000.tfrec'
eval_pattern = root_path + 'data-evaluate_shard-000.tfrec'

#root_path = '/home/aakanksha/Documents/tfrecords/training/'
#train_pattern = root_path+'data-evaluate_shard-000.tfrec'
#eval_pattern = root_path + 'data-evaluate_shard-000.tfrec'

n_classes = 49
block_shape = (32, 32, 32)
batch_size = 8
volume_shape = (256, 256, 256)
n_epochs = None
augment = True
shuffle_buffer_size = 1000
num_parallel_calls = 8
kld_cp = 1
model_path = '/nobackup/users/aakrana/nobrainer_2/nobrainer/models/kwyk/KWYK_weights_CE_50_2.{epoch:03d}-{val_loss:.4f}.h5'
dataset_train = get_dataset(train_pattern,
                            volume_shape, 
                            batch_size,
                            block_shape, 
                            n_classes,
                            one_hot_label=True)
dataset_eval = get_dataset(eval_pattern,
                           volume_shape, 
                           batch_size, 
                           block_shape, 
                           n_classes, 
                           training= False,
                           one_hot_label=True)

steps_per_epoch = nobrainer.dataset.get_steps_per_epoch(
    n_volumes=200,
    volume_shape= volume_shape,#(128,128,128),
    block_shape=block_shape,
    batch_size=batch_size)

steps_per_epoch

validation_steps = nobrainer.dataset.get_steps_per_epoch(
    n_volumes=100,
    volume_shape= volume_shape,#(128,128,128),
    block_shape=block_shape,
    batch_size=batch_size)
callbacks = [tf.keras.callbacks.ModelCheckpoint(model_path)]
strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))
with strategy.scope():
    model = variational_meshnet(n_classes=n_classes, input_shape=(32, 32, 32, 1), filters=96, dropout="concrete", receptive_field=37, is_monte_carlo=True)
    '''
    model.load_weights('//nobackup/users/aakrana/nobrainer_2/nobrainer/models/nobrainer_spikeslab_32iso_weights.h5')
    new_model = tf.keras.Sequential()
    for layer in model.layers[:22]:
        new_model.add(layer)
    import tensorflow_probability as tfp
    for layer in new_model.layers[:22]:
        layer.trainable = False
        
    kernel_posterior_fn = tfp.layers.default_mean_field_normal_fn(
            loc_initializer = tf.keras.initializers.he_normal(),
            loc_regularizer=tf.keras.regularizers.l2(), #None
            untransformed_scale_regularizer=None,
            loc_constraint = tf.keras.constraints.UnitNorm(axis = [0, 1, 2,3]),#None,
            untransformed_scale_constraint=None)
    prior_fn = normal_prior(prior_std=1.0)#prior_fn_for_bayesian()
    kld = None
    new_model.add(tfp.layers.Convolution3DFlipout(filters=115, 
                                kernel_size = 1, 
                                dilation_rate= (1,1,1),
                                padding = 'SAME',
                                activation=tf.nn.softmax, 
                                kernel_prior_fn=prior_fn,
                                kernel_posterior_fn = kernel_posterior_fn,
                                kernel_divergence_fn=kld,
                                name="classification/Kwyk"))
  '''
    model.compile(tf.keras.optimizers.Adam(lr=1e-03),loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[nobrainer.metrics.generalized_dice])
callbacks = [tf.keras.callbacks.ModelCheckpoint(model_path)]        
for e in range(1, 6):
    model.fit(
        dataset_train,
        steps_per_epoch=steps_per_epoch,
        validation_data=dataset_eval,
        validation_steps=validation_steps,
        epochs=e+1,
        initial_epoch=e,callbacks=callbacks)
model.save_weights('weights_kwyk_50_2.hdf5')
