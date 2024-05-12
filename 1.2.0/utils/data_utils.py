from typing import Dict

from nobrainer.dataset import Dataset

from utils.label_mapping import get_label_mapping
from utils.py_utils import main_timer


@main_timer
def load_custom_tfrec(
    config: Dict = None,
    target: str = "train",
):
    if target not in ["train", "eval", "test"]:
        raise ValueError(f"Invalid target: {target}")

    n_classes = config["n_classes"]
    volume_shape = config["volume_shape"]
    block_shape = config["block_shape"]

    label_map = get_label_mapping(n_classes)

    file_pattern = f"/om2/user/hgazula/kwyk_records/kwyk_full/*{target}*"
    volumes = {"train": 10331, "eval": 1148}

    if config["debug"]:
        file_pattern = f"/om2/user/hgazula/nobrainer-data/tfrecords/*{target}*"
        volumes = {"train": 9, "eval": 1}

    dataset = Dataset.from_tfrecords(
        file_pattern=file_pattern,
        volume_shape=volume_shape,
        block_shape=block_shape,
        n_classes=n_classes,
        label_mapping=label_map,
        n_volumes=volumes[target],
    )

    return dataset
