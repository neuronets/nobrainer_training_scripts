import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from pprint import pprint

import nobrainer
import numpy as np
import tensorflow as tf
from nobrainer.dataset import write_multi_resolution
from nobrainer.models import progressivegan as pgan
from nobrainer.processing.generation import ProgressiveGeneration
from nobrainer.volume import adjust_dynamic_range, normalize


def scale(x):
    """Scale data to -1 to 1"""
    return adjust_dynamic_range(normalize(x), [0, 1], [-1, 1])


def main():
    model = pgan(32)
    model[0].summary()
    model[1].summary()


if __name__ == "__main__":
    print(tf.config.list_physical_devices("GPU"))
    # main()
    csv_path = nobrainer.utils.get_data()
    filepaths = nobrainer.io.read_csv(csv_path)

    train_paths = filepaths[:9]

    datasets = write_multi_resolution(
        train_paths,
        tfrecdir="data/generate",
        n_processes=None,
        resolutions=[8, 16, 32],
    )

    # pprint(datasets)

    # Adjust number of epochs
    datasets[8]["epochs"] = 1
    datasets[16]["epochs"] = 1
    datasets[32]["epochs"] = 1
    # datasets[64]["epochs"] = 1

    pprint(datasets)

    # Adjust batch size from the default of 1
    # datasets[8]['batch_size'] = 8
    # datasets[16]['batch_size'] = 8
    # datasets[32]['batch_size'] = 8
    # datasets[64]['batch_size'] = 4

    gen = ProgressiveGeneration()
    gen.fit(datasets, epochs=10, normalizer=None)
