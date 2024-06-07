import glob
import os

import matplotlib.pyplot as plt
import nibabel as nib
import nobrainer
import numpy as np
from nobrainer.dataset import Dataset
from nobrainer.volume import to_blocks

from utils import label_mapping
from utils.label_mapping import get_label_mapping
from utils.plot_utils import get_color_map, plot_tensor_slices


def create_block():
    src_dir = "/om2/user/hgazula/nobrainer-data/datasets"
    dest_dir = "/om2/user/hgazula/nobrainer-data/datasets_block"

    os.makedirs(dest_dir, exist_ok=True)

    images = sorted(glob.glob(os.path.join(src_dir, "*t1*")))
    labels = sorted(glob.glob(os.path.join(src_dir, "*aseg*")))

    for image, label in zip(images, labels):
        image_name = os.path.basename(image)
        label_name = os.path.basename(label)
        print(image_name, label_name)

        image = nib.load(image).get_fdata()
        label = nib.load(label).get_fdata()

        block_id = 220  # hardcoded
        image_block = to_blocks(image.astype(np.float32), block_shape=(32, 32, 32))[
            block_id
        ]
        label_block = to_blocks(label.astype(np.uint16), block_shape=(32, 32, 32))[
            block_id
        ]

        image_block = nib.Nifti1Image(image_block.numpy(), np.eye(4))
        label_block = nib.Nifti1Image(label_block.numpy(), np.eye(4))

        nib.save(image_block, os.path.join(dest_dir, image_name))
        nib.save(label_block, os.path.join(dest_dir, label_name))


def write_csv():
    dest_dir = "/om2/user/hgazula/nobrainer-data/datasets_block"

    images = sorted(glob.glob(os.path.join(dest_dir, "*t1*")))
    labels = sorted(glob.glob(os.path.join(dest_dir, "*aseg*")))

    with open(
        os.path.join("/om2/user/hgazula/nobrainer-data", "filepaths_block.csv"), "w"
    ) as f:
        for image, label in zip(images, labels):
            f.write(f"{image},{label}\n")


def convert_blocks_to_tfrecords():
    filepaths = nobrainer.io.read_csv(
        "/om2/user/hgazula/nobrainer-data/filepaths_block.csv"
    )
    dataset_train, dataset_eval = Dataset.from_files(
        filepaths,
        out_tfrec_dir="/om2/user/hgazula/nobrainer-data/tfrecords_block",
        shard_size=3,
        num_parallel_calls=None,
        n_classes=1,
    )


def plot_blocks(slice_idx=None):
    n_classes = 50
    label_map = label_mapping.get_label_mapping(n_classes)
    labels = sorted(
        glob.glob(os.path.join("/om2/user/hgazula/nobrainer-data/datasets", "*aseg*"))
    )

    y_true = nib.load(labels[-1]).get_fdata()

    u, inv = np.unique(y_true, return_inverse=True)
    y_true = np.array([label_map.get(x, 0) for x in u])[inv].reshape(y_true.shape)

    blocks = to_blocks(y_true, block_shape=(32, 32, 32))

    axis = 0
    os.makedirs(f"eval_slice_plots{axis}", exist_ok=True)
    cmap = get_color_map(n_classes)
    for idx, block in enumerate(blocks):
        if slice_idx is None:
            try:
                plot_tensor_slices(
                    block.numpy(),
                    slice_dim=axis,
                    cmap=cmap,
                    out_name=f"block_label_{idx:03d}.png",
                )
                print(f"Plotting block {idx:03d}")
            except ValueError:
                pass
        elif idx == slice_idx:
            fig = plt.figure()
            plt.imshow(np.arange(50)[None], cmap=cmap)
            plt.show()
            plt.save
        else:
            continue


def plot_blocks_from_record():
    n_classes = 50
    dataset = Dataset.from_tfrecords(
        file_pattern="/om2/user/hgazula/nobrainer-data/tfrecords_block/*eval*",
        volume_shape=(32, 32, 32),
        block_shape=(32, 32, 32),
        n_classes=n_classes,
        label_mapping=get_label_mapping(n_classes),
        n_volumes=1,
    )

    for idx, (_, label) in enumerate(dataset.dataset):
        label_vol = np.squeeze(np.argmax(label.numpy(), axis=-1))
        plot_tensor_slices(
            label_vol,
            slice_dim=0,
            cmap=get_color_map(n_classes),
            out_name=f"example_{idx:03d}.png",
        )


def main():
    create_block()
    write_csv()
    convert_blocks_to_tfrecords()
    plot_blocks_from_record()


if __name__ == "__main__":
    main()
