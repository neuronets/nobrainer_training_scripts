import glob
import os
from multiprocessing import Pool
from pprint import pprint

import nibabel as nib
import numpy as np
import pandas as pd
from icecream import ic

kwyk_label_files = glob.glob("/nese/mit/group/sig/data/kwyk/rawdata/*aseg*.nii.gz")
ic(len(kwyk_label_files))


def get_unique_labels(filepath):
    print(os.path.basename(filepath))
    aseg = nib.load(filepath).get_fdata().astype(np.uint)
    unique_labels = np.unique(aseg).tolist()
    return (filepath, unique_labels, len(unique_labels))


n_procs = len(os.sched_getaffinity(0))
with Pool(processes=n_procs) as pool:
    df_rows = pool.starmap(
        get_unique_labels, [(label_file,) for label_file in kwyk_label_files]
    )

global_label_sizes = set([row[2] for row in df_rows])
labels_list = [row[1] for row in df_rows]

common_labels = set.intersection(*map(set, labels_list))  # common for all volumes
all_labels = set().union(*labels_list)  # all labels across all volumes

ic(global_label_sizes)
ic(all_labels, len(all_labels))
ic(common_labels, len(common_labels))

df = pd.DataFrame(df_rows, columns=["label_file", "unique_labels", "label_size"])
df.to_feather("kwyk_unique_labels.feather")
