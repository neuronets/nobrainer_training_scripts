import collections.abc
import csv
import os
import subprocess
from datetime import datetime

import GPUtil
import psutil

__all__ = [
    "main_timer",
    "_read_csv",
    "get_git_revision_hash",
    "get_git_revision_short_hash",
    "plot_tensor_slices",
    "get_memory_usage",
]


# https://stackoverflow.com/questions/32935232/python-apply-function-to-values-in-nested-dictionary
def map_nested_dicts(ob, func):
    if isinstance(ob, collections.abc.Mapping):
        return {k: map_nested_dicts(v, func) for k, v in ob.items()}
    else:
        return func(ob)


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


def ssh_to_https(ssh_url):
    # Check if the URL is SSH format
    if not ssh_url.startswith("git@"):
        print("Not a valid SSH URL")
        return None

    # Extract the username and host
    parts = ssh_url.split(":")
    if len(parts) != 2:
        print("Invalid SSH URL format")
        return None
    host = parts[0][4:]  # Remove "git@" from the beginning
    path = parts[1].rstrip(".git\n")

    # Construct the HTTPS URL
    https_url = f"https://{host}/{path}"
    return https_url


def get_remote_url(path_to_repo, https=True):
    try:
        # Run the git command to get the remote URL
        result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=path_to_repo,
            capture_output=True,
            text=True,
            check=True,
        )
        remote_url = result.stdout.strip()

        if https and remote_url.startswith("https"):
            return remote_url
        else:
            return ssh_to_https(remote_url)

    except subprocess.CalledProcessError as e:
        print("Error:", e)
        return None


def _read_csv(filepath, skip_header=True, delimiter=","):
    """Return list of tuples from a CSV, where each tuple contains the items
    in a row.
    """
    with open(filepath, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        if skip_header:
            next(reader)
        return [tuple(row) for row in reader]


def get_memory_usage():
    # Get CPU memory usage
    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / 1024 / 1024  # in MB

    # Get GPU memory usage
    gpus = GPUtil.getGPUs()
    gpu_mem = 0
    if gpus:
        gpu = gpus[0]  # Assuming you're using the first GPU
        gpu_mem = gpu.memoryUsed

    return cpu_mem, gpu_mem
