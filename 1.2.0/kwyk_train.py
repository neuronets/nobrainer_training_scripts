# Copyright (c) 2024 MIT
#
# -*- coding:utf-8 -*-
# @Script: kwyk_train.py
# @Author: Harsha
# @Email: hvgazula@users.noreply.github.com
# @Create At: 2024-03-29 09:08:29
# @Last Modified By: Harsha
# @Last Modified At: 2024-05-09 15:16:28
# @Description:
#   1. Code to train bayesian meshnet on kwyk dataset.
#   2. binary segmentation is used in this model.

import ast
import configparser
import os
import sys
from pprint import pprint

from icecream import ic

# ruff: noqa: E402
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from typing import Dict

import create_tfshards
import label_mapping
import nobrainer
import tensorflow as tf
from callbacks_kwyk import TestCallback, get_callbacks
from nobrainer.dataset import Dataset
from nobrainer.models.bayesian_meshnet import variational_meshnet
from nobrainer.processing.segmentation import Segmentation
from nobrainer.volume import standardize

from utils import get_color_map, get_git_revision_short_hash, main_timer

ic.enable()

import collections.abc


# https://stackoverflow.com/questions/32935232/python-apply-function-to-values-in-nested-dictionary
def map_nested_dicts(ob, func):
    if isinstance(ob, collections.abc.Mapping):
        return {k: map_nested_dicts(v, func) for k, v in ob.items()}
    else:
        return func(ob)


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

    file_pattern = f"/om2/user/hgazula/kwyk_records/kwyk_full/*{target}*"
    volumes = {"train": 10331, "eval": 1148}

    if config["debug"]:
        file_pattern = f"/om2/user/hgazula/nobrainer-data/tfrecords/*{target}*"
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
    config.read("/om2/user/hgazula/nobrainer_training_scripts/1.2.0/config.yml")

    config = map_nested_dicts(config._sections, ast.literal_eval)

    pprint(config)

    # basic config
    basic_config = config["basic"]
    model_name = basic_config["model_name"]
    n_classes = basic_config["n_classes"]
    normalize = basic_config["normalize"]

    # training config
    train_config = config["train"]
    n_epochs = train_config["n_epochs"]

    output_dirname = f"{model_name}"

    print(f"Nobrainer version: {nobrainer.__version__}")
    print(f"Git commit hash: {get_git_revision_short_hash()}")

    NUM_GPUS, gpu_names = init_device(flag=False)

    volume_filepaths = create_tfshards.create_filepaths(
        "/nese/mit/group/sig/data/kwyk/rawdata",
        feature_string="orig",
        label_string="aseg",
    )

    if config["basic"]["debug"]:
        volume_filepaths = create_tfshards.create_filepaths(
            "/om2/user/hgazula/nobrainer-data/datasets",
            feature_string="t1",
            label_string="aseg",
        )

    ic(len(volume_filepaths))

    train_list, val_list = create_tfshards.custom_train_val_test_split(
        volume_filepaths,
        train_size=0.90,
        val_size=0.10,
        test_size=0.00,
        random_state=42,
        shuffle=False,
    )

    print("loading data")
    dataset_train, dataset_eval = (
        load_custom_tfrec(config=basic_config, target="eval"),
        load_custom_tfrec(config=basic_config, target="eval"),
    )

    dataset_train = dataset_train.shuffle(NUM_GPUS).batch(NUM_GPUS)
    dataset_eval = dataset_eval.batch(NUM_GPUS)

    if normalize:
        print("normalizing data")
        dataset_train = dataset_train.normalize(normalizer=standardize)
        dataset_eval = dataset_eval.normalize(normalizer=standardize)

    test_callback = TestCallback(
        config,
        val_list,
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

    print("training")
    _ = kwyk.fit(
        dataset_train=dataset_train,
        dataset_validate=dataset_eval,
        epochs=n_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    print("Success")
