# Copyright (c) 2024 MIT
#
# -*- coding:utf-8 -*-
# @Script: brainy_train.py
# @Author: Harsha
# @Email: hvgazula@users.noreply.github.com
# @Create At: 2024-03-29 09:08:29
# @Last Modified By: Harsha
# @Last Modified At: 2024-05-11 06:36:34
# @Description:
#   1. Code to train brainy (unet) on kwyk dataset.
#   2. binary segmentation is used in this model.
#   3. updated to support multi-class segmentation
#   4. added support for plotting predictions

import ast
import configparser
import os
import sys
from argparse import ArgumentParser
from pprint import pprint

from icecream import ic

# ruff: noqa: E402
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from typing import Dict

import nobrainer
import tensorflow as tf
from nobrainer.dataset import Dataset
from nobrainer.models import unet
from nobrainer.processing.segmentation import Segmentation
from nobrainer.volume import standardize

import label_mapping
from callbacks import TestCallback, get_callbacks
from utils import get_git_revision_short_hash, main_timer

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
    parser = ArgumentParser()
    parser.add_argument("config", type=str)

    # If running the code in debug mode
    gettrace = getattr(sys, "gettrace", None)

    if gettrace():
        sys.argv = ["brainy_train.py", "configs/test.yml"]
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

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
    checkpoint_filepath = f"output/{output_dirname}/nobrainer_ckpts/"

    print(f"Nobrainer version: {nobrainer.__version__}")
    print(f"Git commit hash: {get_git_revision_short_hash()}")

    NUM_GPUS, gpu_names = init_device(flag=False)

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

    callbacks = get_callbacks(
        config, output_dirname=output_dirname, gpu_names=gpu_names
    )

    print("creating model")
    bem = Segmentation(
        unet,
        model_args=dict(batchnorm=True),
        multi_gpu=True,
        checkpoint_filepath=checkpoint_filepath,
    )

    print("training")
    bem.fit(
        dataset_train=dataset_train,
        dataset_validate=dataset_eval,
        epochs=n_epochs,
        callbacks=callbacks,
    )

    # # snippet to test resumption. see resume.py
    # # bem = Segmentation.load(checkpoint_filepath)

    # print(checkpoint_filepath)
    # print(os.path.dirname(checkpoint_filepath))

    # bem = Segmentation.init_with_checkpoints(
    #     unet,
    #     model_args=dict(batchnorm=True),
    #     checkpoint_filepath=checkpoint_filepath,
    # )

    print("Success")
