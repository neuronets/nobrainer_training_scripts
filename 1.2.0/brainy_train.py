# Copyright (c) 2024 MIT
#
# -*- coding:utf-8 -*-
# @Script: brainy_train.py
# @Author: Harsha
# @Email: hvgazula@users.noreply.github.com
# @Create At: 2024-03-29 09:08:29
# @Last Modified By: Harsha
# @Last Modified At: 2024-03-29 09:54:22
# @Description: This is description.

import os

# ruff: noqa: E402
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import glob
from nobrainer.dataset import Dataset
import nobrainer

from nobrainer.processing.segmentation import Segmentation
from nobrainer.models import unet
import numpy as np
import nibabel as nib
from datetime import datetime
import copy

NUM_GPUS = len(tf.config.list_physical_devices("GPU"))


def main_timer(func):
    """Decorator to time any function"""

    def function_wrapper(*args, **kwargs):
        start_time = datetime.now()
        # print(f'Start Time: {start_time.strftime("%A %m/%d/%Y %H:%M:%S")}')
        result = func(*args, *kwargs)
        end_time = datetime.now()
        # print(f'End Time: {end_time.strftime("%A %m/%d/%Y %H:%M:%S")}')
        print(
            f"Function: {func.__name__} Total runtime: {end_time - start_time} (HH:MM:SS)"
        )
        return result

    return function_wrapper


def sort_function(item):
    return int(os.path.basename(item).split("_")[1])


def create_filepaths(path_to_data: str, sample: bool = False) -> None:
    """Create filepaths CSV file.

    Args:
        path_to_data (str): Path to data directory.
        sample (bool, optional): Whether to create a sample filepaths CSV. Defaults to False.
    """
    if not path_to_data:
        path_to_data = "/nese/mit/group/sig/data/kwyk/rawdata"

    feature_paths = sorted(
        glob.glob(os.path.join(path_to_data, "*orig*.nii.gz")), key=sort_function
    )
    label_paths = sorted(
        glob.glob(os.path.join(path_to_data, "*aseg*.nii.gz")), key=sort_function
    )

    assert len(feature_paths) == len(
        label_paths
    ), "Mismatch between feature and label paths"

    file_name = "filepaths_sample.csv" if sample else "filepaths.csv"

    with open(file_name, "w") as f:
        for feature, label in zip(feature_paths, label_paths):
            f.write(f"{feature},{label}\n")


@main_timer
def load_sample_files():

    if True:
        csv_path = nobrainer.utils.get_data()
        filepaths = nobrainer.io.read_csv(csv_path)

        dataset_train, dataset_eval = Dataset.from_files(
            filepaths,
            out_tfrec_dir="data/binseg",
            shard_size=3,
            num_parallel_calls=None,
            n_classes=1,
        )
    return dataset_train, dataset_eval


def load_sample_tfrec(target: str = "train"):
    volume_shape = (256, 256, 256)
    block_shape = None

    if target == "train":
        data_pattern = "data/binseg/*train*"
    else:
        data_pattern = "data/binseg/*eval*"

    dataset = Dataset.from_tfrecords(
        file_pattern=data_pattern,
        volume_shape=volume_shape,
        block_shape=block_shape,
        n_volumes=None,
    )

    return dataset


@main_timer
def load_custom_tfrec(target: str = "train"):

    if target == "train":
        data_pattern = "/nese/mit/group/sig/data/kwyk/tfrecords/*train*000*"
    else:
        data_pattern = "/nese/mit/group/sig/data/kwyk/tfrecords/*eval*000*"

    volume_shape = (256, 256, 256)
    block_shape = None

    dataset = Dataset.from_tfrecords(
        file_pattern=data_pattern,
        volume_shape=volume_shape,
        block_shape=block_shape,
        n_volumes=None,
    )

    return dataset


@main_timer
def get_label_count():
    label_count = []
    with open("filepaths.csv", "r") as f:
        lines = f.readlines()[:500]
        for line in lines:
            _, label = line.strip().split(",")
            label_count.append(len(np.unique(nib.load(label).get_fdata())))

    print(set(label_count))


# @main_timer
def main():
    print(tf.config.list_physical_devices("GPU"))

    n_epochs = 10

    model_string = "bem"

    print("loading data")
    if True:
        # run of the following two lines (but not both)
        # dataset_train, dataset_eval = load_sample_files()
        dataset_train, dataset_eval = load_sample_tfrec("train"), load_sample_tfrec(
            "eval"
        )
        save_freq = "epoch"
    else:
        dataset_train, dataset_eval = (
            load_custom_tfrec("train"),
            load_custom_tfrec("eval"),
        )
        save_freq = 100

    dataset_train.repeat(n_epochs).shuffle(1).batch(NUM_GPUS)

    print("creating callbacks")
    callback_model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(f"output/{model_string}/model_ckpts", "model_{epoch:03d}.keras")
    )
    callback_tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=f"output/{model_string}/logs/", histogram_freq=1
    )
    callback_early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=10,
    )
    callback_backup = tf.keras.callbacks.BackupAndRestore(
        backup_dir=f"output/{model_string}/backup", save_freq=save_freq
    )

    callbacks = [
        callback_model_checkpoint,
        callback_tensorboard,
        callback_early_stopping,
        callback_backup,
    ]

    print("creating model")
    bem = Segmentation(
        unet,
        model_args=dict(batchnorm=True),
        multi_gpu=True,
        checkpoint_filepath=f"output/{model_string}/nobrainer_ckpts",
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