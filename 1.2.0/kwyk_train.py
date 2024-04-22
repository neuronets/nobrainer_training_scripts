# Copyright (c) 2024 MIT
#
# -*- coding:utf-8 -*-
# @Script: kwyk_train.py
# @Author: Harsha
# @Email: hvgazula@users.noreply.github.com
# @Create At: 2024-03-29 09:08:29
# @Last Modified By: Harsha
# @Last Modified At: 2024-04-22 15:00:16
# @Description:
#   1. Code to train bayesian meshnet on kwyk dataset.
#   2. binary segmentation is used in this model.
import ast
import configparser
import os
import sys

from icecream import ic

# ruff: noqa: E402
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from typing import Dict

import nobrainer
import tensorflow as tf
from nobrainer.dataset import Dataset
from nobrainer.models.bayesian_meshnet import variational_meshnet
from nobrainer.processing.segmentation import Segmentation
from nobrainer.volume import standardize

import create_tfshards
import label_mapping
from callbacks_kwyk import TestCallback, get_callbacks
from utils import get_color_map, get_git_revision_short_hash, main_timer

ic.enable()


def list_recursive(d, key):
    for k, v in d.items():
        if isinstance(v, dict):
            for found in list_recursive(v, key):
                yield found
        if k == key:
            yield v


def print_config_file(filename):
    try:
        with open(filename, "r") as file:
            config_data = file.read()
            print(config_data)
    except FileNotFoundError:
        print(f"Config file '{filename}' not found.")


@main_timer
def load_custom_tfrec(
    config: Dict = None,
    target: str = "train",
):
    if target not in ["train", "eval", "test"]:
        raise ValueError(f"Invalid target: {target}")

    n_classes = config["n_classes"]

    label_map = label_mapping.get_label_mapping(n_classes)

    file_pattern = f"/om2/scratch/Fri/hgazula/kwyk_tfrecords/*{target}*"
    volumes = {"train": 9757, "eval": 1148, "test": 574}

    # file_pattern = f"/om2/user/hgazula/kwyk_records/kwyk_full/*{target}*"
    # volumes = {"train": None, "eval": None}

    file_pattern = (
        f"/om2/user/hgazula/nobrainer_training_scripts/1.2.0/data/binseg/*{target}*"
    )
    volumes = {"train": 9, "eval": 1}

    dataset = Dataset.from_tfrecords(
        file_pattern=file_pattern,
        volume_shape=config["volume_shape"],
        block_shape=config["block_shape"],
        n_classes=n_classes,
        label_mapping=label_map,
        n_volumes=volumes[target],
    )

    return dataset


def init_device(flag: bool = False):
    gpus = tf.config.list_physical_devices("GPU")
    gpu_names = [item.name for item in gpus]
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    NUM_GPUS = len(gpus)

    if flag and not NUM_GPUS:
        sys.exit("GPU not found")

    return NUM_GPUS, gpu_names


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config.yml")

    # basic config
    basic_config = config["basic"]
    basic_config = {k: ast.literal_eval(v) for k, v in basic_config.items()}

    model_name = basic_config["model_name"]
    n_classes = basic_config["n_classes"]
    normalize = basic_config["normalize"]

    # training config
    train_config = config["training"]
    n_epochs = train_config.getint("n_epochs")

    output_dirname = f"{model_name}_{n_classes}"

    print(f"Nobrainer version: {nobrainer.__version__}")
    print(f"Git commit hash: {get_git_revision_short_hash()}")

    NUM_GPUS, gpu_names = init_device(flag=True)

    volume_filepaths = create_tfshards.create_filepaths(
        "/nese/mit/group/sig/data/kwyk/rawdata"
    )

    train_list, _, test_list = create_tfshards.custom_train_val_test_split(
        volume_filepaths,
        train_size=0.85,
        val_size=0.10,
        test_size=0.05,
        random_state=42,
        shuffle=False,
    )

    ic("loading data")
    dataset_train, dataset_eval = (
        load_custom_tfrec(config=basic_config, target="train"),
        load_custom_tfrec(config=basic_config, target="eval"),
    )

    dataset_train = dataset_train.shuffle(NUM_GPUS).batch(NUM_GPUS)
    dataset_eval = dataset_eval.batch(NUM_GPUS)

    if normalize:
        ic("normalizing data")
        dataset_train = dataset_train.normalize(normalizer=standardize)
        dataset_eval = dataset_eval.normalize(normalizer=standardize)

    test_callback = TestCallback(
        config,
        test_list,
        get_color_map(n_classes),
        f"output/{output_dirname}/predictions",
    )

    callbacks = get_callbacks(output_dirname=output_dirname, gpu_names=gpu_names)
    callbacks.append(test_callback)

    print("creating model")
    kwyk = Segmentation(
        variational_meshnet,
        model_args=dict(
            receptive_field=37,
            filters=96,
            no_examples=len(train_list),
            is_monte_carlo=True,
            dropout="concrete",
        ),
        multi_gpu=True,
        checkpoint_filepath=f"output/{output_dirname}/nobrainer_ckpts",
    )

    ic("training")
    _ = kwyk.fit(
        dataset_train=dataset_train,
        dataset_validate=dataset_eval,
        epochs=n_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    ic("Success")
