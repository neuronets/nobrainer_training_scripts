# Copyright (c) 2024 MIT
#
# -*- coding:utf-8 -*-
# @Script: synthseg_gen-1.py
# @Author: Harsha
# @Email: hvgazula@users.noreply.github.com
# @Create At: 2024-06-19 12:57:53
# @Last Modified By: Harsha
# @Last Modified At: 2024-07-22 07:19:32
# @Description: Synthseg generation in nobrainer.

import os
from datetime import datetime

from utils import py_utils

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from nobrainer.ext.lab2im import utils

if True:
    from nobrainer.processing.brain_generator import BrainGenerator
else:
    from SynthSeg.brain_generator import BrainGenerator

physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

if __name__ == "__main__":
    NUM_EXAMPLES = 5

    # generate an image from the label map.
    brain_generator = BrainGenerator(
        "/net/vast-storage/scratch/vast/gablab/hgazula/nobrainer-data/datasets/sub-10_aparc+aseg.mgz",
        randomise_res=False,
    )

    start = datetime.now()
    count = 1
    while count <= NUM_EXAMPLES:
        im, lab = brain_generator.generate_brain()

        utils.save_volume(
            im,
            brain_generator.aff,
            brain_generator.header,
            "./outputs_tutorial_1/image.nii.gz",
        )
        utils.save_volume(
            lab,
            brain_generator.aff,
            brain_generator.header,
            "./outputs_tutorial_1/labels.nii.gz",
        )

        cpu_usage, gpu_usage = py_utils.get_memory_usage()
        print(
            f"{count:05d} {datetime.now()-start} CPU: {cpu_usage:.2f} MB. GPU: {gpu_usage:.2f} MB"
        )

        count += 1

    print("Success")
