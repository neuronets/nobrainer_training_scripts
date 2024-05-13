import sys

import tensorflow as tf


def init_device(flag: bool = False):
    gpus = tf.config.list_physical_devices("GPU")
    gpu_names = [item.name for item in gpus]
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    NUM_GPUS = len(gpus)

    if flag and not NUM_GPUS:
        sys.exit("GPU not found")

    return NUM_GPUS, gpu_names
