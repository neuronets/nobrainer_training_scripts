# Copyright (c) 2024 MIT
#
# -*- coding:utf-8 -*-
# @Script: synthseg_gen-2.py
# @Author: Harsha
# @Email: hvgazula@users.noreply.github.com
# @Create At: 2024-06-19 13:01:49
# @Last Modified By: Harsha
# @Last Modified At: 2024-06-19 13:26:47
# @Description: For more info, please refer to github.com/bbillot/SynthSeg/scripts/tutorials/2-generation_explained.py.


import os

from nobrainer.ext.lab2im import utils
from nobrainer.processing.brain_generator import BrainGenerator

if __name__ == "__main__":
    n_examples = 5  # number of examples to generate in this script
    result_dir = "./outputs_tutorial_2"  # folder where examples will be saved

    path_label_map = "synth_example_data/training_label_maps"
    generation_labels = "synth_example_data/labels_classes_priors/generation_labels.npy"

    n_neutral_labels = 18

    output_labels = (
        "synth_example_data/labels_classes_priors/synthseg_segmentation_labels.npy"
    )

    n_channels = 1
    target_res = None
    output_shape = 160

    prior_distributions = "uniform"

    generation_classes = (
        "synth_example_data/labels_classes_priors/generation_classes.npy"
    )

    flipping = True  # enable right/left flipping
    scaling_bounds = 0.2  # the scaling coefficients will be sampled from U(1-scaling_bounds; 1+scaling_bounds)
    rotation_bounds = 15  # the rotation angles will be sampled from U(-rotation_bounds; rotation_bounds)
    shearing_bounds = 0.012  # the shearing coefficients will be sampled from U(-shearing_bounds; shearing_bounds)
    translation_bounds = False  # no translation is performed, as this is already modelled by the random cropping
    nonlin_std = (
        4.0  # this controls the maximum elastic deformation (higher = more deformation)
    )
    bias_field_std = (
        0.7  # this controls the maximum bias field corruption (higher = more bias)
    )

    randomise_res = False

    # instantiate BrainGenerator object
    brain_generator = BrainGenerator(
        labels_dir=path_label_map,
        generation_labels=generation_labels,
        n_neutral_labels=n_neutral_labels,
        prior_distributions=prior_distributions,
        generation_classes=generation_classes,
        output_labels=output_labels,
        n_channels=n_channels,
        target_res=target_res,
        output_shape=output_shape,
        flipping=flipping,
        scaling_bounds=scaling_bounds,
        rotation_bounds=rotation_bounds,
        shearing_bounds=shearing_bounds,
        translation_bounds=translation_bounds,
        nonlin_std=nonlin_std,
        bias_field_std=bias_field_std,
        randomise_res=randomise_res,
    )

    for n in range(n_examples):

        # generate new image and corresponding labels
        im, lab = brain_generator.generate_brain()

        # save output image and label map
        utils.save_volume(
            im,
            brain_generator.aff,
            brain_generator.header,
            os.path.join(result_dir, "image_%s.nii.gz" % n),
        )
        utils.save_volume(
            lab,
            brain_generator.aff,
            brain_generator.header,
            os.path.join(result_dir, "labels_%s.nii.gz" % n),
        )
