import os
import random
import resource
from typing import Optional

import nibabel as nib
import numpy as np
import tensorflow as tf
from icecream import ic
from nobrainer.prediction import predict
from nobrainer.volume import standardize
from nvitop.callbacks.keras import GpuStatsLogger

from utils import plot_tensor_slices


class TestCallback(tf.keras.callbacks.Callback):
    def __init__(self, config, test_list, cmap, outdir):
        super(TestCallback, self).__init__()
        self.test_list = test_list
        self.cmap = cmap
        self.slice_dim = np.random.randint(0, 3)
        self.outdir = outdir
        self.config = config

    def on_epoch_end(self, epoch, logs=None):
        print("\nTesting model after epoch {}...".format(epoch + 1))

        self.curr_outdir = os.path.join(self.outdir, f"epoch-{epoch:02d}")
        os.makedirs(self.curr_outdir, exist_ok=True)

        test_samples = random.choices(
            self.test_list, k=self.config["test"]["n_samples"]
        )
        features, labels = zip(*test_samples)

        predictions = predict(
            features,
            self.model,
            self.config["basic"]["block_shape"],
            batch_size=1,  # TODO: batch_size should be configurable when block is implemented
            normalizer=standardize,
            n_samples=self.config["test"]["n_samples"],
            return_variance=False,
            return_entropy=False,
        )

        for label, prediction in zip(labels, predictions):
            subject_id = os.path.basename(label).split(os.extsep, 1)[0]
            ic(subject_id)

            pred_outfile_name = os.path.join(self.curr_outdir, f"{subject_id}_pred.png")
            ic(pred_outfile_name)
            true_outfile_name = os.path.join(self.curr_outdir, f"{subject_id}_true.png")
            ic(true_outfile_name)

            y_true = nib.load(label).get_fdata().astype(np.uint)
            y_pred = prediction.get_fdata().astype(np.uint)
            y_pred[y_true == 0] = 0

            assert y_true.shape == y_pred.shape, f"Shape mismatch at test time"

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
