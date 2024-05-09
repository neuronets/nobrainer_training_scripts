import glob
import os
from pprint import pprint

import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


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

keys = ["sagittal", "axial", "coronal"]

for key in keys:
    frames = []
    for folder in folders:
        frames.append(glob.glob(os.path.join(folder, f"*pred*{key}*"))[0])

    # Create a function to animate the frames
    def animate(i):
        plt.imshow(mpimg.imread(frames[i]))
        plt.tight_layout()
        plt.title(f"Epoch: {i:03d}")

    # Set up the figure
    fig = plt.figure(figsize=(10, 10))
    plt.axis("off")

    # Create the animation
    ani = animation.FuncAnimation(
        fig, animate, frames=len(frames), interval=100, blit=False, repeat=False
    )

    # ani = animation.FuncAnimation(plt.gcf(), animate_new, frames=len(frames),
    #                      interval=(2000.0/nframes), blit=False, repeat=False)
    ani.save(
        os.path.join(working_dir, experiment_name, f"{key}.gif"), writer="imagemagick"
    )

    # Uncomment the line below if you want to save the animation to a file
    # ani.save('animation.mp4', writer='ffmpeg', fps=10)

    # # Show the animation
    # plt.show()
