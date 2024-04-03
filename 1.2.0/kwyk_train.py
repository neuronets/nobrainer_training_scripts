# Copyright (c) 2024 MIT
#
# -*- coding:utf-8 -*-
# @Script: kwyk_train.py
# @Author: Harsha
# @Email: hvgazula@users.noreply.github.com
# @Create At: 2024-03-29 09:08:29
# @Last Modified By: Harsha
# @Last Modified At: 2024-04-03 08:22:43
# @Description:
#   1. Code to train bayesian meshnet on kwyk dataset.
#   2. binary segmentation is used in this model.

import os
import sys

# ruff: noqa: E402
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from typing import Dict, Optional

import label_mapping
import nobrainer
import tensorflow as tf
from nobrainer.dataset import Dataset
from nobrainer.models.bayesian_meshnet import variational_meshnet
from nobrainer.processing.segmentation import Segmentation
from nvitop.callbacks.keras import GpuStatsLogger

from utils import get_git_revision_short_hash, main_timer

# tf.data.experimental.enable_debug_mode()

print(f"Nobrainer version: {nobrainer.__version__}")
print(f"Git commit hash: {get_git_revision_short_hash()}")


@main_timer
def load_custom_tfrec(
    target: str = "train", n_classes: int = 1, label_mapping: Dict = None
):
    if target == "train":
        # data_pattern = "/nese/mit/group/sig/data/kwyk/tfrecords/*train*"
        data_pattern = "/om2/scratch/Fri/hgazula/kwyk_full/*train*"
    else:
        # data_pattern = "/nese/mit/group/sig/data/kwyk/tfrecords/*eval*"
        data_pattern = "/om2/scratch/Fri/hgazula/kwyk_full/*eval*"

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

    callbacks = [
        # callback_gpustats,  # gpu stats callback should be placed before tboard/csvlogger callback
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

    n_epochs = 20
    n_classes = 50
    output_dirname = "kwyk_mc_test"

    label_map = label_mapping.get_label_mapping(n_classes)

    print("loading data")
    dataset_train, dataset_eval = (
        load_custom_tfrec(target="train", n_classes=n_classes, label_mapping=label_map),
        load_custom_tfrec(target="eval", n_classes=n_classes, label_mapping=label_map),
    )

    dataset_train.shuffle(NUM_GPUS).batch(NUM_GPUS)
    dataset_eval.batch(NUM_GPUS)

    callbacks = get_callbacks(output_dirname=output_dirname, gpu_names=gpu_names)

    print("creating model")
    kwyk = Segmentation(
        variational_meshnet,
        model_args=dict(no_examples=9200, filters=21),
        multi_gpu=True,
        checkpoint_filepath=f"output/{output_dirname}/nobrainer_ckpts",
    )

    print("training")
    _ = kwyk.fit(
        dataset_train=dataset_train,
        dataset_validate=dataset_eval,
        epochs=n_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    print("Success")


if __name__ == "__main__":
    main()
