import os
import random
import resource
from typing import Optional

import nibabel as nib
import numpy as np
import tensorflow as tf
from icecream import ic
from nvitop.callbacks.keras import GpuStatsLogger

from utils import get_color_map, plot_tensor_slices


class TestCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_list, cmap, outdir):
        super(TestCallback, self).__init__()
        self.test_list = test_list
        self.outdir = outdir
        self.cmap = cmap
        self.slice_dim = np.random.randint(0, 3)

        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        print("\nTesting model after epoch {}...".format(epoch + 1))

        # Select a random sample from the test dataset
        for _ in range(5):
            random_index = random.randint(0, len(self.test_list) - 1)
            random_sample = self.test_list[random_index]

            subject_id = os.path.basename(random_sample[0]).split(os.extsep, 1)[0]
            ic(subject_id)

            pred_outfile_name = os.path.join(
                self.outdir, f"{subject_id}_epoch-{epoch:02d}_pred.png"
            )
            ic(pred_outfile_name)
            true_outfile_name = os.path.join(
                self.outdir, f"{subject_id}_epoch-{epoch:02d}_true.png"
            )
            ic(true_outfile_name)

            x_data = nib.load(random_sample[0]).get_fdata().astype(np.float32)
            y_true = nib.load(random_sample[1]).get_fdata().astype(np.float32)

            prediction = self.model.predict(x_data[None, ..., None], batch_size=1)
            y_pred = np.squeeze(prediction).argmax(-1)

            plot_tensor_slices(
                y_pred,
                slice_dim=self.slice_dim,
                cmap=self.cmap,
                crop_percentile=10,
                out_name=pred_outfile_name,
            )
            plot_tensor_slices(
                y_true,
                slice_dim=self.slice_dim,
                cmap=self.cmap,
                crop_percentile=10,
                out_name=true_outfile_name,
            )


class MemoryLoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.memory_usage = []

    def on_test_batch_begin(self, batch, logs=None):
        max_used_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(
            "Start of batch {}: max memory usage: {} MB".format(batch, max_used_memory)
        )

    def on_test_batch_end(self, batch, logs=None):
        max_used_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print("End of batch {}: max memory usage: {} MB".format(batch, max_used_memory))

    def on_epoch_end(self, epoch, logs=None):
        max_used_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(f"\nmemory usage after end of epoch hook: {max_used_memory}")
        if epoch % 10 == 0:
            self.memory_usage.append(max_used_memory)
            print(self.memory_usage)


def get_callbacks(
    output_dirname: str = "test",
    save_freq: int = 250,
    gpu_names: Optional[list[str]] = None,
):
    print("creating callbacks")

    callback_model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(f"output/{output_dirname}/model_ckpts", "model_{epoch:02d}.keras")
    )
    callback_tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=f"output/{output_dirname}/logs/", histogram_freq=1
    )
    callback_early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=10,
    )
    callback_backup = tf.keras.callbacks.BackupAndRestore(
        backup_dir=f"output/{output_dirname}/backup", save_freq=save_freq
    )
    callback_gpustats = GpuStatsLogger(gpu_names)

    memory_logger_callback = MemoryLoggerCallback()

    callbacks = [
        # callback_gpustats,  # gpu stats callback should be placed before tboard/csvlogger callback
        # memory_logger_callback,
        callback_model_checkpoint,
        callback_tensorboard,
        callback_early_stopping,
        callback_backup,
    ]

    return callbacks
