# Copyright (c) 2024 MIT
#
# -*- coding:utf-8 -*-
# @Script: brainy_train.py
# @Author: Harsha
# @Email: hvgazula@users.noreply.github.com
# @Create At: 2024-03-29 09:08:29
# @Last Modified By: Harsha
# @Last Modified At: 2024-04-19 19:15:59
# @Description:
#   1. Code to train brainy (unet) on kwyk dataset.
#   2. binary segmentation is used in this model.
#   3. updated to support multi-class segmentation
#   4. added support for plotting predictions

import os
import random
import resource
import sys

import numpy as np
from icecream import ic

# ruff: noqa: E402
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from typing import Dict, Optional

import nibabel as nib
import nobrainer
import tensorflow as tf
from nobrainer.dataset import Dataset
from nobrainer.models import unet
from nobrainer.processing.segmentation import Segmentation
from nvitop.callbacks.keras import GpuStatsLogger

import create_tfshards
import label_mapping
from utils import (
    get_color_map,
    get_git_revision_short_hash,
    main_timer,
    plot_tensor_slices,
)

# tf.data.experimental.enable_debug_mode()

print(f"Nobrainer version: {nobrainer.__version__}")
print(f"Git commit hash: {get_git_revision_short_hash()}")


@main_timer
def load_custom_tfrec(
    target: str = "train", n_classes: int = 1, label_mapping: Dict = None
):
    if target == "train":
        # data_pattern = "/nese/mit/group/sig/data/kwyk/tfrecords/*train*"
        data_pattern = "/om2/user/hgazula/kwyk_records/kwyk_eighth/*train*00*"
        data_pattern = (
            "/om2/user/hgazula/nobrainer_training_scripts/1.2.0/data/binseg/*train*"
        )
        # data_pattern = (
        #     "/om2/user/hgazula/nobrainer_training_scripts/1.2.0/kwyk_10/*train-000*"
        # )
    elif target == "eval":
        # data_pattern = "/nese/mit/group/sig/data/kwyk/tfrecords/*eval*"
        data_pattern = "/om2/user/hgazula/kwyk_records/kwyk_quarter/*eval*"
        data_pattern = (
            "/om2/user/hgazula/nobrainer_training_scripts/1.2.0/data/binseg/*eval*000*"
        )
        # data_pattern = "/om2/user/hgazula/nobrainer_training_scripts/1.2.0/kwyk_25/*eval*"
    else:
        data_pattern = "/om2/scratch/Fri/hgazula/kwyk_tfrecords/*test*"

    volume_shape = (256, 256, 256)
    block_shape = None

    dataset = Dataset.from_tfrecords(
        file_pattern=data_pattern,
        volume_shape=volume_shape,
        block_shape=block_shape,
        n_classes=n_classes,
        label_mapping=label_mapping,
    )

    return dataset


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


def main():
    gpus = tf.config.list_physical_devices("GPU")
    gpu_names = [item.name for item in gpus]
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    NUM_GPUS = len(gpus)

    if not NUM_GPUS:
        sys.exit("GPU not found")

    n_epochs = 10
    n_classes = 50
    output_dirname = "brainy_mc50"

    label_map = label_mapping.get_label_mapping(n_classes)

    print("loading data")
    dataset_train, dataset_eval = (
        load_custom_tfrec(target="train", n_classes=n_classes, label_mapping=label_map),
        load_custom_tfrec(target="eval", n_classes=n_classes, label_mapping=label_map),
    )

    volume_filepaths = create_tfshards.create_filepaths(
        "/nese/mit/group/sig/data/kwyk/rawdata"
    )
    _, _, test_list = create_tfshards.custom_train_val_test_split(
        volume_filepaths,
        train_size=0.85,
        val_size=0.10,
        test_size=0.05,
        random_state=42,
        shuffle=False,
    )

    test_callback = TestCallback(
        test_list, get_color_map(n_classes), f"output/{output_dirname}/predictions"
    )

    dataset_train = dataset_train.repeat(4).shuffle(NUM_GPUS).batch(NUM_GPUS)
    dataset_eval = dataset_eval.batch(NUM_GPUS)

    callbacks = get_callbacks(output_dirname=output_dirname, gpu_names=gpu_names)
    callbacks.append(test_callback)

    print("creating model")
    bem = Segmentation(
        unet,
        model_args=dict(batchnorm=True),
        multi_gpu=True,
        checkpoint_filepath=f"output/{output_dirname}/nobrainer_ckpts",
    )

    print("training")
    _ = bem.fit(
        dataset_train=dataset_train,
        dataset_validate=dataset_eval,
        epochs=n_epochs,
        callbacks=callbacks,
    )

    print("Success")


if __name__ == "__main__":
    main()
