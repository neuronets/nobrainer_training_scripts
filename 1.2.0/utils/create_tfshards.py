# Copyright (c) 2024 MIT
#
# -*- coding:utf-8 -*-
# @Script: create_tfshards.py
# @Author: Harsha
# @Email: hvgazula@users.noreply.github.com
# @Create At: 2024-03-29 20:19:47
# @Last Modified By: Harsha
# @Last Modified At: 2024-04-26 09:07:10
# @Description: Create tfrecords of kwyk data

import glob
import logging
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from typing import List

import nobrainer
import numpy as np
import tensorflow as tf
from nobrainer.dataset import Dataset
from sklearn.model_selection import train_test_split

_TFRECORDS_DTYPE = "float32"

# tf.config.run_functions_eagerly(True)


def setup_logging(log_file="script.log"):
    """Set up logging configuration."""
    logging.basicConfig(
        filename=log_file,
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def custom_train_val_test_split(
    *arrays,
    train_size=0.6,
    val_size=0.2,
    test_size=0.2,
    shuffle=False,
    random_state=None,
):
    """Split the dataset into three sets with custom percentages."""

    if sum([train_size, val_size, test_size]) != 1:
        raise ValueError("train_size + val_size + test_size should be 1")

    if val_size + test_size > 1:
        raise ValueError("val_size + test_size should be smaller than 1")

    if len(arrays) == 0:
        raise ValueError("At least one array required as input")

    if len(arrays) == 1:
        arrays = arrays[0]

    if not isinstance(arrays, (list, tuple)):
        arrays = [arrays]

    if random_state is not None:
        np.random.seed(random_state)

    # arrays = [np.asarray(a) for a in arrays]

    if train_size + val_size + test_size > 1:
        raise ValueError("train_size + val_size + test_size should be smaller than 1")

    if train_size < 0 or val_size < 0 or test_size < 0:
        raise ValueError(
            "All values in (train_size, val_size, test_size) should be positive"
        )

    if train_size + val_size + test_size == 0:
        raise ValueError(
            "All values in (train_size, val_size, test_size) should be non-zero"
        )

    if test_size == 0 or val_size == 0:
        if (train_size + test_size == 1) or (train_size + val_size == 1):
            splitting = train_test_split(
                arrays,
                train_size=train_size,
                shuffle=shuffle,
                random_state=random_state,
            )
            return splitting
        else:
            raise ValueError(
                "train_size + test|val_size should be equal to 1 when val|test_size is 0"
            )

    output = train_test_split(
        arrays, test_size=test_size, shuffle=shuffle, random_state=random_state
    )

    train_val_split_ratio = val_size / (1 - test_size)
    if len(output) == 4:
        X_train_temp, X_test, y_train_temp, y_test = output
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_temp,
            y_train_temp,
            test_size=train_val_split_ratio,
            random_state=random_state,
        )

        assert len(X_train) + len(X_val) + len(X_test) == len(arrays[0])
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        X_train_temp, X_test = output
        X_train, X_val = train_test_split(
            X_train_temp,
            test_size=train_val_split_ratio,
            shuffle=shuffle,
            random_state=random_state,
        )
        return X_train, X_val, X_test


def sort_function(item):
    try:
        key = int(os.path.basename(item).split("_")[1])
    except ValueError as e:
        key = 0
    return key


def create_filepaths(
    path_to_data: str, feature_string: str = "orig", label_string: str = "aseg"
) -> List:
    """Create filepaths CSV file.

    Args:
        path_to_data: Path to data directory.
        sample: Whether to create a sample filepaths CSV.
    """
    if not path_to_data:
        path_to_data = "/nese/mit/group/sig/data/kwyk/rawdata"

    feature_paths = sorted(
        glob.glob(os.path.join(path_to_data, f"*{feature_string}*")),
        key=sort_function,
    )
    label_paths = sorted(
        glob.glob(os.path.join(path_to_data, f"*{label_string}*")),
        key=sort_function,
    )

    assert len(feature_paths) == len(
        label_paths
    ), "Mismatch between feature and label paths"

    return list(zip(feature_paths, label_paths))


def get_record_size(path, target="eval"):
    if target not in ["train", "eval", "test"]:
        raise ValueError(f"Invalid target: {target}")

    file_pattern = sorted(glob.glob(f"{path}*{target}*"))

    if len(file_pattern) == 0:
        return 0

    first_file = file_pattern[0]
    last_file = file_pattern[-1]

    first_file_idx = os.path.splitext(os.path.basename(first_file))[0].split("-")[-1]
    last_file_idx = os.path.splitext(os.path.basename(last_file))[0].split("-")[-1]

    num_shards = int(last_file_idx) - int(first_file_idx) + 1

    assert num_shards == len(file_pattern)

    print(f"Num of {target} shards: {num_shards}")

    print(f"{target} first file", end=" ")
    first_dataset = Dataset.from_tfrecords(
        file_pattern=first_file,
        volume_shape=(256, 256, 256),
        block_shape=None,
        n_classes=1,
    )

    print(f"{target} last file", end=" ")
    last_dataset = Dataset.from_tfrecords(
        file_pattern=last_file,
        volume_shape=(256, 256, 256),
        block_shape=None,
        n_classes=1,
    )

    n_vols_in_first = len([0 for _ in first_dataset.dataset])
    n_vols_in_last = len([0 for _ in last_dataset.dataset])

    total_vols = (n_vols_in_first * (num_shards - 1)) + n_vols_in_last
    print(f"Total number of volumes in {target}: {total_vols}")

    return total_vols


def create_kwyk_tfrecords(
    examples_per_shard=25,
    output_dir=os.getcwd(),
    train_size=0.6,
    val_size=0.2,
    test_size=0.2,
):
    volume_filepaths = create_filepaths("/nese/mit/group/sig/data/kwyk/rawdata")
    n_volumes = len(volume_filepaths)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logging.info("Creating tfrecords")
    logging.info(
        f"Sharding {n_volumes} volumes into {examples_per_shard} examples per shard"
    )
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Train, val, test (percent): {train_size, val_size, test_size}")

    output_list = custom_train_val_test_split(
        volume_filepaths,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=42,
        shuffle=False,
    )

    zipup = list(zip(["train", "eval", "test"], output_list))
    logging.info([f"{key} count: {len(value)}" for key, value in zipup])

    for shard_type, item in zipup:
        nobrainer.tfrecord.write(
            features_labels=item,
            filename_template=f"{output_dir}/{shard_type}" + "-{shard:03d}.tfrec",
            examples_per_shard=examples_per_shard,
            compressed=True,
        )

    logging.info("successfully created tfrecords")


def parse_example(serialized):
    """Parse one example from a TFRecord file made with Nobrainer.

    Parameters
    ----------
    serialized: str, serialized proto message.

    Returns
    -------
    Tuple of two tensors. If `scalar_labels` is `False`, both tensors have shape
    `volume_shape`. Otherwise, the first tensor has shape `volume_shape`, and the
    second is a scalar tensor.
    """
    volume_shape = (256, 256, 256)
    scalar_labels = None

    features = {
        "feature/shape": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "feature/value": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "label/value": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "label/rank": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
    }
    e = tf.io.parse_single_example(serialized=serialized, features=features)
    x = tf.io.decode_raw(e["feature/value"], _TFRECORDS_DTYPE)
    y = tf.io.decode_raw(e["label/value"], _TFRECORDS_DTYPE)
    # TODO: this line does not work. The shape cannot be determined
    # dynamically... for now.
    # xshape = tf.cast(
    #     tf.io.decode_raw(e["feature/shape"], _TFRECORDS_DTYPE), tf.int32)
    x = tf.reshape(x, shape=volume_shape)
    if not scalar_labels:
        y = tf.reshape(y, shape=volume_shape)
    else:
        y = tf.reshape(y, shape=[1])
    return x, y


def read_tfrecord(file_pattern="train*"):
    volume_shape = (256, 256, 256)
    block_shape = None

    dataset = nobrainer.dataset.Dataset.from_tfrecords(
        file_pattern=file_pattern,
        volume_shape=volume_shape,
        block_shape=block_shape,
        n_volumes=None,
    )

    # dataset = tf.data.Dataset.list_files("eval*", shuffle=False)
    # hello = tf.data.TFRecordDataset(dataset, compression_type="GZIP")
    # parse_fn = nobrainer.tfrecord.parse_example_fn(
    #     volume_shape=volume_shape, scalar_labels=False
    # )
    # # hello1 = hello.map(parse_fn)

    # # for item in hello1:
    # #     print(item)

    # for item in hello:
    #     parse_example(item)

    print(len([0 for _ in dataset.dataset]))
    return dataset


def count_volumes_in_shards(path_to_dir):
    total_vol_count = 0
    for target in ["train", "eval", "test"]:
        total_vol_count += get_record_size(path_to_dir, target)
    print(total_vol_count)


if __name__ == "__main__":
    setup_logging(os.path.splitext(__file__)[0] + ".log")
    create_kwyk_tfrecords(
        examples_per_shard=20,
        output_dir="/om2/user/hgazula/kwyk_records/kwyk_full",
        train_size=0.90,
        val_size=0.10,
        test_size=0.0,
    )
    # read_tfrecord(file_pattern="/om2/user/hgazula/kwyk_records/kwyk_eighth/*eval*000*")

    # count_volumes_in_shards("/om2/user/hgazula/kwyk_records/kwyk_full")
