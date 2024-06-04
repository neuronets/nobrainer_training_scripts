import glob
import os

import nibabel as nib
import nobrainer
import numpy as np
from nobrainer.dataset import Dataset

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

        image_block = image[111 : 111 + 32, 111 : 111 + 32, 111 : 111 + 32]
        label_block = label[111 : 111 + 32, 111 : 111 + 32, 111 : 111 + 32]

        image_block = image_block.astype(np.float32)
        label_block = label_block.astype(np.int16)

        image_block = nib.Nifti1Image(image_block, np.eye(4))
        label_block = nib.Nifti1Image(label_block, np.eye(4))

        nib.save(image_block, os.path.join(dest_dir, image_name))
        nib.save(label_block, os.path.join(dest_dir, label_name))


def write_csv():
    dest_dir = "/om2/user/hgazula/nobrainer-data/datasets_block"

    images = sorted(glob.glob(os.path.join(dest_dir, "*t1*")))
    labels = sorted(glob.glob(os.path.join(dest_dir, "*aseg*")))

    with open(os.path.join(dest_dir, "block.csv"), "w") as f:
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


def plot_blocks():
    n_classes = 50
    label = sorted(
        glob.glob(
            os.path.join("/om2/user/hgazula/nobrainer-data/datasets_block", "*aseg*")
        )
    )

    label = nib.load(label[0]).get_fdata()
    print(label.shape)
    cmap = get_color_map(n_classes)
    plot_tensor_slices(label, slice_dim=0, cmap=cmap, out_name="label_32.png")


def main():
    # create_block()
    # write_csv()
    # convert_blocks_to_tfrecords()
    plot_blocks()


if __name__ == "__main__":
    main()
