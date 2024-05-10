# Copyright (c) 2024 MIT
#
# -*- coding:utf-8 -*-
# @Script: test.py
# @Author: Harsha
# @Email: hvgazula@users.noreply.github.com
# @Create At: 2024-05-10 10:38:05
# @Last Modified By: Harsha
# @Last Modified At: 2024-05-10 10:38:31
# @Description:
#  1. sample code to demonstrate resumption
#  2. this same trick doesn't work in nobrainer API


import keras
import numpy as np

verbose = 1


class InterruptingCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 4:
            raise RuntimeError("Interrupting!")


callback = keras.callbacks.BackupAndRestore(
    backup_dir="backup", delete_checkpoint=False
)
model = keras.models.Sequential([keras.layers.Dense(10)])
model.compile(keras.optimizers.SGD(), loss="mse")
try:
    model.fit(
        np.arange(100).reshape(5, 20),
        np.zeros(5),
        epochs=10,
        batch_size=1,
        callbacks=[callback, InterruptingCallback()],
        verbose=verbose,
    )
except Exception as e:
    print(e)
history = model.fit(
    np.arange(100).reshape(5, 20),
    np.zeros(5),
    epochs=10,
    batch_size=1,
    callbacks=[callback],
    verbose=verbose,
)
print(len(history.history["loss"]))

model.save("hello")
model = keras.models.load_model("hello")
history = model.fit(
    np.arange(100).reshape(5, 20),
    np.zeros(5),
    epochs=15,
    batch_size=1,
    callbacks=[callback],
    verbose=verbose,
)
print(len(history.history["loss"]))

history = model.fit(
    np.arange(100).reshape(5, 20),
    np.zeros(5),
    epochs=16,
    batch_size=1,
    callbacks=[callback],
    verbose=verbose,
)
print(len(history.history["loss"]))
