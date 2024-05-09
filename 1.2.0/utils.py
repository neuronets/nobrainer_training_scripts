import csv
import subprocess
from datetime import datetime

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ext.mindboggle.labels import extract_numbers_names_colors

__all__ = [
    "main_timer",
    "_read_csv",
    "get_git_revision_hash",
    "get_git_revision_short_hash",
    "plot_tensor_slices",
]


def main_timer(func):
    """Decorator to time any function"""

    def function_wrapper(*args, **kwargs):
        start_time = datetime.now()
        # print(f'Start Time: {start_time.strftime("%A %m/%d/%Y %H:%M:%S")}')
        result = func(*args, **kwargs)
        end_time = datetime.now()
        # print(f'End Time: {end_time.strftime("%A %m/%d/%Y %H:%M:%S")}')
        print(
            f"Function: {func.__name__} Total runtime: {end_time - start_time} (HH:MM:SS)"
        )
        return result

    return function_wrapper


# https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script
def get_git_revision_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def get_git_revision_short_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )


def _read_csv(filepath, skip_header=True, delimiter=","):
    """Return list of tuples from a CSV, where each tuple contains the items
    in a row.
    """
    with open(filepath, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        if skip_header:
            next(reader)
        return [tuple(row) for row in reader]


def crop_tensor(tensor, percentile=10):
    # Create a copy of the tensor to avoid modifying the original
    data_for_processing = tensor.copy()

    # Thresholding (assuming background has very low values compared to the head)
    threshold = np.percentile(data_for_processing, percentile)
    data_for_processing[data_for_processing < threshold] = 0

    # Find the bounding box around the head (non-zero region) in the filtered data
    indices = np.nonzero(data_for_processing)
    min_z, max_z = np.min(indices[0]), np.max(indices[0])
    min_y, max_y = np.min(indices[1]), np.max(indices[1])
    min_x, max_x = np.min(indices[2]), np.max(indices[2])

    # Crop the original tensor using the bounding box from the filtered data
    cropped_tensor = tensor[min_z : max_z + 1, min_y : max_y + 1, min_x : max_x + 1]

    return cropped_tensor, [min_z, max_z, min_x, max_x, min_y, max_y]


def plot_tensor_slices(
    tensor,
    slice_dim=0,
    cmap="viridis",
    crop_percentile=10,
    out_name=None,
    crop_dims=None,
):
    # Crop the tensor
    if not crop_dims:
        cropped_tensor, [min_z, max_z, min_x, max_x, min_y, max_y] = crop_tensor(
            tensor, percentile=crop_percentile
        )
        crop_dims = [min_z, max_z, min_x, max_x, min_y, max_y]
    else:
        [min_z, max_z, min_x, max_x, min_y, max_y] = crop_dims
        cropped_tensor = tensor[min_z : max_z + 1, min_y : max_y + 1, min_x : max_x + 1]

    # Determine the dimensions of the cropped tensor
    dim0, dim1, dim2 = cropped_tensor.shape

    # Determine the slicing dimensions based on the specified slice_dim
    if slice_dim == 0:
        num_slices = dim0
        slice_shape = (dim1, dim2)
    elif slice_dim == 1:
        num_slices = dim1
        slice_shape = (dim0, dim2)
    elif slice_dim == 2:
        num_slices = dim2
        slice_shape = (dim0, dim1)
    else:
        raise ValueError("Invalid slice_dim. Must be 0, 1, or 2.")

    # Calculate the grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_slices)))

    # Create a larger matrix to hold the slices
    R = np.zeros((grid_size * slice_shape[0], grid_size * slice_shape[1]))

    # Iterate over the slices and place them in the larger matrix
    for i in range(grid_size):
        for j in range(grid_size):
            slice_index = i * grid_size + j
            if slice_index < num_slices:
                if slice_dim == 0:
                    slice_data = cropped_tensor[slice_index, :, :]
                elif slice_dim == 1:
                    slice_data = cropped_tensor[:, slice_index, :]
                else:  # slice_dim == 2
                    slice_data = cropped_tensor[:, :, slice_index]
                R[
                    i * slice_shape[0] : (i + 1) * slice_shape[0],
                    j * slice_shape[1] : (j + 1) * slice_shape[1],
                ] = slice_data

    # Plot the larger matrix
    plt.figure(figsize=(10, 10))
    plt.imshow(R, cmap=cmap, interpolation="nearest")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_name, dpi=600)
    # plt.show()

    return crop_dims


def get_color_map(n_classes=50):
    fs_number, fs_names, fs_colors = extract_numbers_names_colors(
        "/om2/user/hgazula/freesurfer/FreeSurferColorLUT.txt"
    )

    if n_classes not in [50, 115, 6]:
        raise ValueError("n_classes must be 6, 50, or 115")

    if n_classes == 115:
        df = pd.read_csv(
            "/om2/user/hgazula/nobrainer_training_scripts/csv-files/115-class-mapping.csv",
            header="infer",
        )
    if n_classes == 50:
        df = pd.read_csv(
            "/om2/user/hgazula/nobrainer_training_scripts/csv-files/50-class-mapping.csv",
            header="infer",
            index_col=0,
        )
    if n_classes == 6:
        df = pd.read_csv(
            "/om2/user/hgazula/nobrainer_training_scripts/csv-files/6-class-mapping.csv",
            header="infer",
            index_col=0,
        )

    df["colors"] = df["original"].apply(lambda x: fs_colors[fs_number.index(x)])
    df = df.drop_duplicates(subset="new")

    my_colors = df.colors.tolist()
    my_colors[0] = [255, 255, 255]  # replacing background with white color
    cmap = mcolors.ListedColormap(np.array(my_colors) / 255)

    assert len(cmap.colors) == n_classes, "Incorrect number of colors in colormap"

    return cmap
