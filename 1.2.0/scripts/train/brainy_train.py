# Copyright (c) 2024 MIT
#
# -*- coding:utf-8 -*-
# @Script: brainy_train.py
# @Author: Harsha
# @Email: hvgazula@users.noreply.github.com
# @Create At: 2024-03-29 09:08:29
# @Last Modified By: Harsha
# @Last Modified At: 2024-05-13 10:22:33
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

import nobrainer
from nobrainer.models import unet
from nobrainer.processing.segmentation import Segmentation
from nobrainer.volume import standardize

from utils.callbacks import get_callbacks
from utils.data_utils import load_custom_tfrec
from utils.py_utils import get_git_revision_short_hash, get_remote_url, map_nested_dicts
from utils.tf_utils import init_device

ic.enable()


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
    output_dir = basic_config["model_name"]
    normalize = basic_config["normalize"]

    # training config
    train_config = config["train"]
    n_epochs = train_config["n_epochs"]

    checkpoint_filepath = f"output/{output_dir}/model_chkpts/" + "{epoch:02d}"

    print(f"Nobrainer version: {nobrainer.__version__}")
    print(f"URL: {get_remote_url(os.getcwd())}/tree/{get_git_revision_short_hash()}")

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

    print("creating model")
    model = Segmentation.init_with_checkpoints(
        unet,
        model_args=dict(batchnorm=True),
        checkpoint_filepath=checkpoint_filepath,
    )

    callbacks = get_callbacks(config, model, output_dir=output_dir, gpu_names=gpu_names)

    print("training")
    model.fit(
        dataset_train=dataset_train,
        dataset_validate=dataset_eval,
        epochs=n_epochs,
        callbacks=callbacks,
    )

    print("Success")
