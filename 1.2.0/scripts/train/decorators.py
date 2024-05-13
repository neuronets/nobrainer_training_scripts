# Copyright (c) 2024 MIT
#
# -*- coding:utf-8 -*-
# @Script: decorators.py
# @Author: Harsha
# @Email: hvgazula@users.noreply.github.com
# @Create At: 2024-03-29 09:08:29
# @Last Modified By: Harsha
# @Last Modified At: 2024-05-13 07:06:03
# @Description:
#   1. Code to train brainy (unet) on kwyk dataset.
#   2. binary segmentation is used in this model.
#   3. updated to support multi-class segmentation
#   4. added support for plotting predictions

import ast
import configparser
import functools
import os
import sys
from argparse import ArgumentParser
from pprint import pprint

from icecream import ic

# ruff: noqa: E402
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import nobrainer
from nobrainer.models import unet
from nobrainer.models.bayesian_meshnet import variational_meshnet
from nobrainer.processing.segmentation import Segmentation
from nobrainer.volume import standardize

from utils.data_utils import load_custom_tfrec
from utils.callbacks import get_callbacks
from utils.py_utils import get_git_revision_short_hash, map_nested_dicts
from utils.tf_utils import init_device
from utils.data_utils import load_custom_tfrec

ic.enable()


def argument_parser():
    parser = ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("config", type=str)

    # If running the code in debug mode
    gettrace = getattr(sys, "gettrace", None)

    if gettrace():
        sys.argv = ["train.py", "brainy", "configs/test.yml"]

    args = parser.parse_args()
    return args


def train(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args = argument_parser()

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

        # args that need to be passed for model building
        kwargs["checkpoint_filepath"] = checkpoint_filepath
        kwargs["dataset_train"] = dataset_train

        print("creating model")
        model = func(**kwargs)

        callbacks = get_callbacks(
            config, model, output_dir=output_dir, gpu_names=gpu_names
        )

        print("training")
        result = model.fit(
            dataset_train=dataset_train,
            dataset_validate=dataset_eval,
            epochs=n_epochs,
            callbacks=callbacks,
        )

        print("Success")
        return result

    return wrapper


@train
def brainy(*args, **kwargs):
    checkpoint_filepath = kwargs["checkpoint_filepath"]

    model = Segmentation.init_with_checkpoints(
        unet,
        model_args=dict(batchnorm=True),
        checkpoint_filepath=checkpoint_filepath,
    )
    return model


@train
def kwyk(*args, **kwargs):
    dataset_train = kwargs["dataset_train"]
    checkpoint_filepath = kwargs["checkpoint_filepath"]

    model = Segmentation.init_with_checkpoints(
        variational_meshnet,
        model_args=dict(
            receptive_field=37,
            filters=96,
            no_examples=dataset_train.get_steps_per_epoch() * dataset_train.batch_size,
            is_monte_carlo=True,
            dropout="concrete",
        ),
        checkpoint_filepath=checkpoint_filepath,
    )
    return model


if __name__ == "__main__":
    args = argument_parser()
    model_dict = {"brainy": brainy, "kwyk": kwyk}
    model_dict[args.model](args)
