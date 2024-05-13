# Copyright (c) 2024 MIT
#
# -*- coding:utf-8 -*-
# @Script: label_mapping.py
# @Author: Harsha
# @Email: hvgazula@users.noreply.github.com
# @Create At: 2024-04-03 06:37:51
# @Last Modified By: Harsha
# @Last Modified At: 2024-04-12 16:37:59
# @Description:
# 1. Script to map freesurfer labels to 0-n_classes-1 (segmentation labels).
# 2. Adapted from the label_mapping.py script written by previous authors.


import os
from typing import Dict

import numpy as np
import pandas as pd

PRJCT_DIR = "/om2/user/hgazula/nobrainer_training_scripts/"
LABEL_FILES = {
    6: "6-class-mapping.csv",
    50: "50-class-mapping.csv",
    115: "115-class-mapping.csv",
}


def get_label_mapping(n_classes: int) -> Dict[int, int]:
    """
    Load label mapping for n_classes.

    Args:
        n_classes (int): Number of classes to load label mapping for.

    Returns:
        label_mapping: Dictionary mapping original labels to new labels.

    Raises:
        NotImplementedError: If n_classes is not 50, 115, or 6.

    The function loads a label mapping file based on the number of classes provided.
    The conversion is from freesurfer labels to 0-n_classes-1.
    The unknown class (label 0) is optionally removed from the mapping.

    TODO:
        1. Add option to remove the unknown class from the mapping.
        2. Add path to label mapping file.
    """
    if n_classes not in [1, 2, 6, 50, 115]:
        raise NotImplementedError

    if n_classes in [1, 2]:
        return None

    label_file = os.path.join(PRJCT_DIR, "csv-files", LABEL_FILES[n_classes])
    print(f"Using label mapping file: {label_file}")
    print(f"Conversion into {n_classes} segmentation classes from freesurfer labels")

    df = pd.read_csv(label_file, header=0)
    df = df[["original", "new"]].astype(np.int32)
    df = df.iloc[1:, :]
    label_mapping = dict(zip(df["original"], df["new"]))

    return label_mapping


if __name__ == "__main__":
    label_mapping = get_label_mapping(6)
