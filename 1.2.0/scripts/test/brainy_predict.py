# Copyright (c) 2024 MIT
#
# -*- coding:utf-8 -*-
# @Script: brainy_predict.py
# @Author: Harsha
# @Email: hvgazula@users.noreply.github.com
# @Create At: 2024-04-02 07:05:48
# @Last Modified By: Harsha
# @Last Modified At: 2024-04-20 08:02:48
# @Description:
#   1. Code to predict brainy (unet) on kwyk dataset.

import os
import random

import create_tfshards
import matplotlib.pyplot as plt
from nilearn import plotting
from nobrainer.processing.segmentation import Segmentation
from nobrainer.volume import standardize

DATA_DIR = "/nese/mit/group/sig/data/kwyk/rawdata"
PRJCT_DIR = "/om2/user/hgazula/nobrainer_training_scripts"
FILEPATHS_CSV = os.path.join(PRJCT_DIR, "1.2.0", "filepaths.csv")
SAVED_MODEL = os.path.join(PRJCT_DIR, "1.2.0", "output/brainy_mc50/nobrainer_ckpts")


volume_filepaths = create_tfshards.create_filepaths(
    "/nese/mit/group/sig/data/kwyk/rawdata"
)

*_, test_list = create_tfshards.custom_train_val_test_split(
    volume_filepaths,
    train_size=0.85,
    val_size=0.10,
    test_size=0.05,
    random_state=42,
    shuffle=False,
)

# randomly sample 10 volumes
eval_list = random.sample(test_list, 3)

segmentation = Segmentation.load(SAVED_MODEL)

# iterate over dataset and predict
for feature_path, label_path in eval_list[:1]:
    label_pred = segmentation.predict(feature_path, normalizer=standardize)

    fig = plt.figure(figsize=(12, 6))
    plotting.plot_roi(
        label_pred,
        bg_img=feature_path,
        cut_coords=(0, 10, -21),
        alpha=0.4,
        vmin=0,
        vmax=5,
        figure=fig,
    )

    # save figure
    fig.savefig("output.png", dpi=600)
    # TODO:
    # 1. proper renaming of output
    # 2. overlay label map on feature volume instead
