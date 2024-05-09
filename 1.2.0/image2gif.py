import imageio.v3 as iio
import glob
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from argparse import ArgumentParser
from pathlib import Path


parser = ArgumentParser()
parser.add_argument("--key", type=str, default="sagittal")
args = parser.parse_args()

key = args.key
print(f"key: {key}")


def sort_key(item):
    return int(item.split("-")[-1])


working_dir = "/om2/user/hgazula/nobrainer_training_scripts/output"
experiment_name = "kwyk_test_2024_05_08_T0_50"

print(f"Working directory: {working_dir}")
print(f"Loading experiment: {experiment_name}")

folders = sorted(
    glob.glob(os.path.join(working_dir, experiment_name, "predictions/epoch-*")),
    key=sort_key,
)

print("Number of epochs: ", len(folders))

# for key in keys[:1]:
#     frames = []
#     for folder in folders:
#         frames.append(glob.glob(os.path.join(folder, f"*pred*{key}*"))[0])

#     images = []
#     for filename in frames[-5:]:
#         print(filename)
#         images.append(iio.imread(filename))

#     np_images = np.stack(images, axis=0)
#     iio.imwrite(f"{key}_iio.gif", np_images, loop=0)
#     optimize(f"{key}_iio.gif")


frames = []
for folder in folders:
    frames.append(glob.glob(os.path.join(folder, f"*pred*{key}*"))[0])

images = []
for idx, filename in enumerate(frames):
    print(Path(*Path(filename).parts[-2:]))

    i = Image.open(filename)
    Im = ImageDraw.Draw(i)
    mf = ImageFont.truetype(
        "/usr/share/fonts/liberation/LiberationMono-Regular.ttf", 175
    )
    # Add Text to an image
    Im.text((150, 150), f"Epoch: {idx:03d}", (0, 0, 0), font=mf)
    images.append(i)

images[0].save(
    os.path.join(working_dir, experiment_name, f"{key}.gif"),
    save_all=True,
    append_images=images[1:],
    optimize=True,
    loop=0,
    duration=500,
)
