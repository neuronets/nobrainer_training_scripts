import os
from typing import Dict
import numpy as np
import pandas as pd

LABEL_FILES = {
    50: "50-class-mapping.csv",
    115: "115-class-mapping.csv",
    6: "6-class-mapping.csv",
}


def label_mapping(n_classes: int) -> Dict[int, int]:
    """
    Load label mapping for n_classes.

    Args:
        n_classes: Number of classes to load label mapping for.

    Returns:
        label_mapping: Dictionary mapping original labels to new labels.

    Raises:
        NotImplementedError: If n_classes is not 50, 115, or 6.
    """
    label_file = os.path.join(
        os.path.dirname(__file__), "csv_files", LABEL_FILES[n_classes]
    )
    print(f"Using label mapping file: {label_file}")

    print(
        f"Conversion into {n_classes} segmentation classes from freesurfer labels to 0-{n_classes-1}"
    )
    if n_classes == 50:
        tmp = pd.read_csv(label_file, header=0, usecols=[1, 2], dtype=np.int32)
        tmp = tmp.iloc[1:, :]  # removing the unknown class
    elif n_classes == 115:
        tmp = pd.read_csv(label_file, header=0, usecols=[0, 1], dtype=np.int32)
    elif n_classes == 6:
        tmp = pd.read_csv(label_file, header=0, usecols=[1, 2], dtype=np.int32)
    else:
        raise NotImplementedError(f"n_classes must be 50, 115, or 6, not {n_classes}")

    label_mapping = dict(zip(tmp["original"], tmp["new"]))
    del tmp

    return label_mapping


if __name__ == "__main__":
    label_mapping(50)
