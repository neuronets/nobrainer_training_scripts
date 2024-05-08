import glob
import os

import tensorflow as tf
from nobrainer import losses, metrics
from nobrainer.models import unet
from original_unet_3d_tf import OriginalUnetTF
from tensorflow import keras


def get_compiled_model():
    inputs = keras.Input(shape=(1, 256, 256, 256))
    # x = keras.layers.Dense(256, activation="relu")(inputs)
    # x = keras.layers.Dense(256, activation="relu")(x)
    # outputs = keras.layers.Dense(10)(x)
    # model = keras.Model(inputs, outputs)

    model = OriginalUnetTF(1, 50)

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=losses.dice,
        # metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model


def get_brain_dataset():
    path_to_rawdata = "/om/user/hodaraja/kwyk/rawdata"
    path_to_tfrecords = "/om/user/hodaraja/kwyk/tfrecords"

    train_data_pattern = os.path.join(path_to_tfrecords, "*train*")
    eval_data_pattern = os.path.join(path_to_tfrecords, "*eval*")

    train_records = sorted(glob.glob(train_data_pattern))
    eval_records = sorted(glob.glob(eval_data_pattern))

    print(len(train_records), len(eval_records))

    train_dataset = tf.data.TFRecordDataset(
        train_records,
        compression_type=None,
        buffer_size=None,
        num_parallel_reads=None,
        name=None,
    )

    val_dataset = tf.data.TFRecordDataset(
        eval_records,
        compression_type=None,
        buffer_size=None,
        num_parallel_reads=None,
        name=None,
    )

    return train_dataset, val_dataset


# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

# Open a strategy scope.
with strategy.scope():
    # Everything that creates variables should be under the strategy scope.
    # In general this is only model construction & `compile()`.
    model = get_compiled_model()

# Train the model on all available devices.
train_dataset, val_dataset = get_brain_dataset()
model.fit(train_dataset, epochs=2, validation_data=val_dataset)

# Test the model on all available devices.
model.evaluate(val_dataset)
