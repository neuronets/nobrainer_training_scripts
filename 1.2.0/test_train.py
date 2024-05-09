# Copyright (c) 2024 MIT
#
# -*- coding:utf-8 -*-
# @Script: test_train.py
# @Author: Harsha
# @Email: hvgazula@users.noreply.github.com
# @Create At: 2024-03-29 09:08:29
# @Last Modified By: Harsha
# @Last Modified At: 2024-04-26 07:59:42
# @Description: This is description.

import os
import sys

# ruff: noqa: E402
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import glob

import nibabel as nib
import nobrainer
import numpy as np
import tensorflow as tf
from nobrainer.dataset import Dataset
from nobrainer.models import unet
from nobrainer.processing.segmentation import Segmentation
from nobrainer.volume import standardize

from label_mapping import get_label_mapping
from utils import main_timer

# tf.data.experimental.enable_debug_mode()


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
def load_sample_files(n_classes: int = 1, label_mapping: dict = None):
    if True:
        csv_path = nobrainer.utils.get_data()
        filepaths = nobrainer.io.read_csv(csv_path)

        dataset_train, dataset_eval = Dataset.from_files(
            filepaths,
            out_tfrec_dir="data/binseg",
            shard_size=3,
            num_parallel_calls=None,
            n_classes=n_classes,
            block_shape=None,
            label_mapping=label_mapping,
        )
    return dataset_train, dataset_eval


def load_sample_tfrec(target: str = "train"):
    volume_shape = (256, 256, 256)
    # block_shape = (128, 128, 128)
    block_shape = None

    if target == "train":
        data_pattern = "data/binseg/*train*000*"
    else:
        data_pattern = "data/binseg/*eval*000*"

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
        data_pattern = "/nese/mit/group/sig/data/kwyk/tfrecords/*train*"
        data_pattern = "/om2/scratch/Fri/hgazula/kwyk_full/*train*"
    else:
        data_pattern = "/nese/mit/group/sig/data/kwyk/tfrecords/*eval*"
        data_pattern = "/om2/scratch/Fri/hgazula/kwyk_full/*eval*"

    volume_shape = (256, 256, 256)
    block_shape = None

    dataset = Dataset.from_tfrecords(
        file_pattern=data_pattern,
        volume_shape=volume_shape,
        block_shape=block_shape,
    )

    return dataset


@main_timer
def get_label_count():
    label_count = []
    with open("filepaths.csv", "r") as f:
        lines = f.readlines()
        for line in lines:
            _, label = line.strip().split(",")
            label_count.append(len(np.unique(nib.load(label).get_fdata())))

    print(set(label_count))


# @main_timer
def main():
    NUM_GPUS = len(tf.config.list_physical_devices("GPU"))

    if not NUM_GPUS:
        sys.exit("GPU not found")

    n_epochs = 1
    n_classes = 50
    label_mapping_dict = get_label_mapping(n_classes)

    print("loading data")
    if True:
        # run one of the following two lines (but not both)
        # the second line won't succeed unless the first one is run at least once

        # dataset_train, dataset_eval = load_sample_files(
        #     n_classes=n_classes, label_mapping=label_mapping_dict
        # )
        dataset_train, dataset_eval = (
            load_sample_tfrec("train"),
            load_sample_tfrec("eval"),
        )
        model_string = "test6"
        save_freq = "epoch"
    else:
        dataset_train, dataset_eval = (
            load_custom_tfrec("train"),
            load_custom_tfrec("eval"),
        )
        model_string = "test"
        save_freq = 250

    dataset_train = dataset_train.shuffle(NUM_GPUS).batch(NUM_GPUS)
    dataset_eval = dataset_eval.batch(NUM_GPUS)

    # callbacks = get_callbacks(output_dirname=output_dirname, gpu_names=gpu_names)
    # callbacks.append(test_callback)

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
        # checkpoint_filepath=f"output/{model_string}/nobrainer_ckpts",
    )

    # batch_size = 1
    # while True:
    #     try:
    #         train_copy = dataset_train
    #         train_copy.dataset = dataset_train.dataset.take(batch_size)
    #         train_copy.repeat(2).batch(batch_size)
    #         _ = bem.fit(
    #             dataset_train=train_copy,
    #         )  # TODO: add a flag for summary
    #         batch_size *= 2
    #     except tf.errors.ResourceExhaustedError as e:
    #         batch_size //= 2
    #         break
    #     except ValueError as e:
    #         batch_size //= 2
    #         break

    # tf.keras.backend.clear_session()
    # try:
    #     import gc

    #     gc.collect()
    # except Exception:
    #     pass
    # return batch_size

    print("Actual training")
    # dataset_train.batch(batch_size)
    _ = bem.fit(
        dataset_train=dataset_train,
        dataset_validate=dataset_eval,
        epochs=n_epochs,
        callbacks=callbacks,
    )

    print("Success")

    image_path = "/nese/mit/group/sig/data/kwyk/rawdata/pac_0_orig.nii.gz"
    out = bem.predict(image_path, normalizer=standardize)
    print(out.shape)


if __name__ == "__main__":
    main()
