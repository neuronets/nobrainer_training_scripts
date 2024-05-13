# Copyright (c) 2024 MIT
#
# -*- coding:utf-8 -*-
# @Script: examples.py
# @Author: Harsha
# @Email: hvgazula@users.noreply.github.com
# @Create At: 2024-03-29 14:45:04
# @Last Modified By: Harsha
# @Last Modified At: 2024-04-23 12:26:24
# @Description: Simple tensorflow code for kwyk dataset (without nobrainer API).

import glob
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pdb
from datetime import datetime

import nibabel as nib
import numpy as np
import tensorflow as tf
from nobrainer.models import unet
from nobrainer.dataset import Dataset

DATA_DIR = "/nese/mit/group/sig/data/kwyk/rawdata"
PRJCT_DIR = os.getcwd()

AUTOTUNE = tf.data.AUTOTUNE


# tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()

physical_devices = tf.config.list_physical_devices("GPU")
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


def label_is_scalar(self):
    return tf.experimental.numpy.isscalar(self)


def main():
    print(datetime.now())
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            tf.random.uniform(
                shape=(500, 256, 256, 256), minval=0, maxval=255, dtype=tf.int32
            ),
            tf.random.uniform(
                shape=(500, 256, 256, 256), minval=0, maxval=50, dtype=tf.int32
            ),
        )
    )
    print(datetime.now())

    dataset.map(lambda x, y: (x, tf.expand_dims(binarize(y), -1)), num_parallel_calls=8)

    print(datetime.now())
    print((_labels_all_scalar([y for _, y in dataset])))
    print(datetime.now())

    print("QED")


def apply_isscalar(ds):
    return ds.map(lambda _, y: tf.experimental.numpy.isscalar(y))


def main1():
    dataset = tf.data.Dataset.from_tensor_slices(
        (np.arange(100000), np.arange(100000, 200000))
    )

    print(datetime.now())
    dataset1 = dataset.apply(apply_isscalar)
    print(datetime.now())
    dataset2 = dataset.map(
        lambda _, y: tf.experimental.numpy.isscalar(y),
        deterministic=False,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    print(tf.math.reduce_all(list(dataset1.as_numpy_iterator())).numpy())
    print(tf.math.reduce_all(list(dataset2.as_numpy_iterator())).numpy())


def load_nifti_file(file_path):
    img = nib.load(file_path)
    data = img.get_fdata()
    return data


def load_label_map(file_path):
    img = nib.load(file_path)
    data = img.get_fdata()
    return data


def process_data(img, label):
    # Your processing logic here
    # For example, you might want to normalize the image data
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img, label


def create_dataset(data_dir, batch_size=32):
    # files = os.listdir(data_dir)
    # file_paths = [
    #     os.path.join(data_dir, file)
    #     for file in files
    #     if file.endswith(".nii") or file.endswith(".nii.gz")
    # ]

    file_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*orig*")))

    dataset = tf.data.Dataset.from_tensor_slices(file_paths)

    def load_and_process_data(file_path):
        file_path = str(file_path.numpy())
        img = tf.numpy_function(load_nifti_file, [file_path], tf.float32)

        # Assuming label file name is same as image file name with '_label' appended
        label_file_path = file_path.replace("orig", "aseg")
        label = tf.numpy_function(load_label_map, [label_file_path], tf.int32)

        img, label = tf.py_function(process_data, [img, label], [tf.float32, tf.int32])
        return img, label

    dataset = dataset.map(load_and_process_data)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


train_dataset = create_dataset(DATA_DIR, batch_size=32)

# def main():
#     NUM_GPUS = len(tf.config.list_physical_devices("GPU"))

#     path_features = os.path.join(DATA_DIR, "*orig*")
#     path_labels = os.path.join(DATA_DIR, "*aseg*")

#     path_features = sorted(glob.glob(path_features))[:10]
#     path_labels = sorted(glob.glob(path_labels))[:10]

#     features = tf.data.Dataset.from_tensor_slices(path_features)
#     labels = tf.data.Dataset.from_tensor_slices(path_labels)

#     dataset = tf.data.Dataset.zip((features, labels))

#     dataset = dataset.map(load_and_process_data)
#     dataset = dataset.batch(NUM_GPUS)
#     dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

#     # for x, y in dataset:
#     #     print(x, y)
#     #     exit()

#     print("SUCCESS")


def add_ten(example, label):
    pdb.set_trace()
    example_plus_ten = example + 10  # Breakpoint here.
    return example_plus_ten, label


def map_debug():
    """example code to debug map function"""
    examples = [10, 20, 30, 40, 50, 60, 70, 80]
    labels = [0, 0, 1, 1, 1, 1, 0, 0]

    examples_dataset = tf.data.Dataset.from_tensor_slices(examples)
    labels_dataset = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((examples_dataset, labels_dataset))
    dataset = dataset.map(
        map_func=lambda example, label: tf.py_function(
            func=add_ten, inp=[example, label], Tout=[tf.int32, tf.int32]
        )
    )
    dataset = dataset.batch(2)
    example_and_label = next(iter(dataset))


def batch_behavior():
    """simple example to test batch behavior"""
    dataset = tf.data.Dataset.from_tensor_slices([8, 3, 0, 8, 2, 1, 2, 3])
    dataset = dataset.shuffle(buffer_size=2).repeat(2)
    dataset = dataset.batch(3, drop_remainder=True)

    for element in dataset:
        print(element.numpy())


def write_example():
    features = tf.random.uniform(shape=(10, 5, 5), minval=0, maxval=255, dtype=tf.int32)
    labels = tf.random.uniform(shape=(10, 5, 5), minval=0, maxval=5, dtype=tf.int32)

    def create_example(idx, features, labels):
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "features": tf.train.Feature(
                        int64_list=tf.train.Int64List(
                            value=tf.reshape(features[idx], [-1])
                        )
                    ),
                    "labels": tf.train.Feature(
                        int64_list=tf.train.Int64List(
                            value=tf.reshape(labels[idx], [-1])
                        )
                    ),
                }
            )
        )
        return example

    with tf.io.TFRecordWriter("example.tfrecord") as writer:
        for i in range(10):
            example = create_example(i, features, labels)
            writer.write(example.SerializeToString())


def read_example():
    features = {
        "features": tf.io.FixedLenFeature(shape=[25], dtype=tf.int64),
        "labels": tf.io.FixedLenFeature(shape=[25], dtype=tf.int64),
    }

    for serialized_example in tf.data.TFRecordDataset("example.tfrecord"):
        example = tf.io.parse_single_example(
            serialized=serialized_example, features=features
        )
        print(example)


dataset = tf.dataset((feature, vlabel, percent, blah))
dataset.filter(lambda feature, vlabel, percent, blah: blah < 60)
# def read_example_test():
#     for example in tf.data.TFRecordDataset("example.tfrecord"):
#         example = tf.train.Example.FromString(example.numpy())
#         print(example)


def write_sequence_example():
    num_volumes = 14
    vols_per_shard = 3

    shard_vol_lists = np.array_split(np.arange(num_volumes), vols_per_shard)
    num_shards = len(shard_vol_lists)

    features = tf.random.uniform(
        shape=(num_volumes, 10, 10), minval=0, maxval=255, dtype=tf.int32
    )
    labels = tf.random.uniform(
        shape=(num_volumes, 10, 10), minval=0, maxval=5, dtype=tf.int32
    )

    for shard_idx, shard_vol_list in enumerate(shard_vol_lists):
        sequence_example = tf.train.SequenceExample()
        sequence_example.context.feature["length"].int64_list.value.append(
            len(shard_vol_list)
        )

        for vol_idx in shard_vol_list:
            sequence_example.feature_lists.feature_list[
                "features"
            ].feature.add().int64_list.value.extend(features[vol_idx].numpy().flatten())
            sequence_example.feature_lists.feature_list[
                "labels"
            ].feature.add().int64_list.value.extend(labels[vol_idx].numpy().flatten())

        with tf.io.TFRecordWriter(
            f"sequence_example_{shard_idx:02d}-of-{num_shards:02d}.tfrecord"
        ) as writer:
            writer.write(sequence_example.SerializeToString())


def read_sequence_example():
    shards = glob.glob("./1.2.0/sequence_example*.tfrecord")

    # Define features
    context_features = {
        "length": tf.io.FixedLenFeature([], dtype=tf.int64),
    }
    sequence_features = {
        # 'labels': tf.FixedLenSequenceFeature([], dtype=tf.float32),
        "features": tf.io.VarLenFeature(dtype=tf.int64),
        "labels": tf.io.VarLenFeature(dtype=tf.int64),
    }

    for shard in shards[:1]:
        dataset = tf.data.TFRecordDataset(shard)
        for idx, serialized_example in enumerate(dataset):
            context_data, sequence_data = tf.io.parse_single_sequence_example(
                serialized_example,
                context_features=context_features,
                sequence_features=sequence_features,
            )
            print(context_data["length"].numpy())


def model_test(model_cls, n_classes, input_shape, kwds={}):
    """Tests for models."""
    x = 10 * np.random.random(input_shape)
    y = np.random.choice([True, False], input_shape)

    # Assume every model class has n_classes and input_shape arguments.
    model = model_cls(n_classes=n_classes, input_shape=input_shape[1:], **kwds)
    model.compile(tf.optimizers.Adam(), "binary_crossentropy")
    model.fit(x, y)

    actual_output = model.predict(x)
    assert actual_output.shape == x.shape[:-1] + (n_classes,)


def test_unet():
    model_test(unet, n_classes=1, input_shape=(1, 32, 32, 32, 1))


if __name__ == "__main__":
    # model = unet(n_classes=5, input_shape=(256, 256, 256, 1))
    # model.summary()
    # batch_behavior()
    # write_sequence_example()
    # read_sequence_example()

    # write_example()
    # read_example()
    file_pattern = "/om2/user/hgazula/kwyk_records/kwyk_full/*eval*000*"
    train_dataset = Dataset.from_tfrecords(
        file_pattern=file_pattern,
        volume_shape=(256, 256, 256),
        block_shape=None,
        n_classes=1,
    )

    # print(len([0 for _ in train.dataset]))
