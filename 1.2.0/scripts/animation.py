import glob
import os
from argparse import ArgumentParser
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# import imageio
# import imageio.v3 as iio
# import matplotlib.animation as animation
# import matplotlib.image as mpimg
# import matplotlib.pyplot as plt
# import numpy as np
# from pprint import pprint


parser = ArgumentParser()
parser.add_argument("--key", type=str, default="sagittal")
args = parser.parse_args()

key = args.key
print(f"key: {key}")


def sort_key(item):
    return int(item.split("-")[-1])


working_dir = "/om2/user/hgazula/nobrainer_training_scripts/1.2.0/output/20240610"
experiment_name = "kwyk_t0"

print(f"Working directory: {working_dir}")
print(f"Loading experiment: {experiment_name}")

folders = sorted(
    glob.glob(os.path.join(working_dir, experiment_name, "predictions/epoch-*")),
    key=sort_key,
)

print("Number of epochs: ", len(folders))

# keys = ["sagittal", "axial", "coronal"]

# for key in keys:
#     frames = []
#     for folder in folders:
#         frames.append(glob.glob(os.path.join(folder, f"*pred*{key}*"))[0])

#     # Create a function to animate the frames
#     def animate(i):
#         plt.imshow(mpimg.imread(frames[i]))
#         plt.tight_layout()
#         plt.title(f"Epoch: {i:03d}")

#     # Set up the figure
#     fig = plt.figure(figsize=(10, 10))
#     plt.axis("off")

#     # Create the animation
#     ani = animation.FuncAnimation(
#         fig, animate, frames=len(frames), interval=100, blit=False, repeat=False
#     )

#     # ani = animation.FuncAnimation(plt.gcf(), animate_new, frames=len(frames),
#     #                      interval=(2000.0/nframes), blit=False, repeat=False)
#     ani.save(
#         os.path.join(working_dir, experiment_name, f"{key}.gif"), writer="imagemagick"
#     )

#     # Uncomment the line below if you want to save the animation to a file
#     # ani.save('animation.mp4', writer='ffmpeg', fps=10)

#     # # Show the animation
#     # plt.show()

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
    if idx % 10 != 0:
        continue

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
    os.path.join(working_dir, experiment_name, f"kwyk_t0_{key}.gif"),
    save_all=True,
    append_images=images[1:],
    optimize=True,
    loop=0,
    duration=500,
)

# imageio.mimwrite("out.png", images, duration=4)
