import os
import random
import resource
from typing import Optional

import nibabel as nib
import numpy as np
import tensorflow as tf
from icecream import ic
from nobrainer.prediction import predict
from nobrainer.processing.checkpoint import CheckpointTracker
from nobrainer.volume import standardize
from nvitop.callbacks.keras import GpuStatsLogger

from utils import create_tfshards, label_mapping
from utils.plot_utils import get_color_map, plot_tensor_slices


class TestCallback(tf.keras.callbacks.Callback):
    def __init__(self, config, test_list, outdir):
        super(TestCallback, self).__init__()
        self.test_list = test_list
        self.outdir = outdir
        self.config = config
        self.n_classes = self.config["basic"]["n_classes"]  # 6 or 50 or 115
        self.label_map = label_mapping.get_label_mapping(self.n_classes)
        self.normalizer = standardize if self.config["basic"]["normalize"] else None
        self.cmap = get_color_map(self.n_classes)

    def on_epoch_end(self, epoch, logs=None):
        if self.model.stop_training:
            return

        print("\nTesting model after epoch {}...".format(epoch + 1))

        self.curr_outdir = os.path.join(self.outdir, f"epoch-{epoch + 1:02d}")
        os.makedirs(self.curr_outdir, exist_ok=True)

        test_samples = random.choices(
            self.test_list, k=self.config["test"]["n_samples"]
        )
        features, labels = zip(*test_samples)

        predictions = predict(
            features,
            self.model,
            self.config["basic"]["block_shape"],
            batch_size=1,  # TODO: batch_size should be configurable when block is implemented
            normalizer=self.normalizer,
            n_samples=self.config["test"]["n_samples"],
            return_variance=False,
            return_entropy=False,
        )

        for label, prediction in zip(labels, predictions):
            subject_id = os.path.basename(label).split(os.extsep, 1)[0]
            ic(subject_id)

            y_true = nib.load(label).get_fdata().astype(np.uint)
            y_pred = prediction.get_fdata().astype(np.uint)
            y_pred[y_true == 0] = 0

            assert y_true.shape == y_pred.shape, f"Shape mismatch at test time"
            assert (
                len(np.unique(y_pred)) <= len(np.unique(y_true)) == self.n_classes
            ) == True, "something's wrong"

            if self.n_classes in [1, 2]:
                y_true = (y_true > 0).astype(np.uint8)  # binarize
            else:
                u, inv = np.unique(y_true, return_inverse=True)
                y_true = np.array([self.label_map.get(x, 0) for x in u])[inv].reshape(
                    y_true.shape
                )
            assert len(np.unique(y_true)) == self.n_classes, "something's wrong"

            for slice_dim, dim_name in zip(range(3), ["sagittal", "axial", "coronal"]):
                pred_outfile_name = os.path.join(
                    self.curr_outdir, f"{subject_id}_pred_{dim_name}.png"
                )
                true_outfile_name = os.path.join(
                    self.curr_outdir, f"{subject_id}_true_{dim_name}.png"
                )

                crop_dims = plot_tensor_slices(
                    y_true,
                    slice_dim=slice_dim,
                    cmap=self.cmap,
                    crop_percentile=10,
                    out_name=true_outfile_name,
                    crop_dims=None,
                )

                plot_tensor_slices(
                    y_pred,
                    slice_dim=slice_dim,
                    cmap=self.cmap,
                    crop_percentile=10,
                    out_name=pred_outfile_name,
                    crop_dims=crop_dims,
                )


class MemoryLoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.memory_usage = []

    def on_test_batch_begin(self, batch, logs=None):
        max_used_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(
            "Start of batch {}: max memory usage: {} MB".format(batch, max_used_memory)
        )

    def on_test_batch_end(self, batch, logs=None):
        max_used_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print("End of batch {}: max memory usage: {} MB".format(batch, max_used_memory))

    def on_epoch_end(self, epoch, logs=None):
        max_used_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(f"\nmemory usage after end of epoch hook: {max_used_memory}")
        if epoch % 10 == 0:
            self.memory_usage.append(max_used_memory)
            print(self.memory_usage)


def get_validation_file_list(config):
    volume_filepaths = create_tfshards.create_filepaths(
        "/nese/mit/group/sig/data/kwyk/rawdata",
        feature_string="orig",
        label_string="aseg",
    )

    if config["basic"]["debug"]:
        volume_filepaths = create_tfshards.create_filepaths(
            "/om2/user/hgazula/nobrainer-data/datasets",
            feature_string="t1",
            label_string="aseg",
        )

    ic(len(volume_filepaths))

    _, val_list = create_tfshards.custom_train_val_test_split(
        volume_filepaths,
        train_size=0.90,
        val_size=0.10,
        test_size=0.00,
        random_state=42,
        shuffle=False,
    )

    return val_list


def get_callbacks(
    config,
    model,
    output_dir: str = "test",
    gpu_names: Optional[list[str]] = None,
):
    print("creating callbacks")

    callback_tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=f"output/{output_dir}/logs/", histogram_freq=1
    )
    callback_early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=10,
    )

    callback_backup = tf.keras.callbacks.BackupAndRestore(
        backup_dir=f"output/{output_dir}/backup",
        save_freq=config["train"]["save_freq"],
    )

    if gpu_names:
        callback_gpustats = GpuStatsLogger(gpu_names)

    callback_mem_logger = MemoryLoggerCallback()

    callback_nanterminate = tf.keras.callbacks.TerminateOnNaN()

    callback_plotting = TestCallback(
        config,
        get_validation_file_list(config),
        f"output/{output_dir}/predictions",
    )

    callback_best_checkpoint = CheckpointTracker(
        model, f"output/{output_dir}/best_model", save_best_only=True
    )

    callbacks = [
        # callback_gpustats,  # gpu stats callback should be placed before tboard/csvlogger callback
        # callback_mem_logger,
        callback_tensorboard,
        # callback_early_stopping,
        # callback_backup,  # turning off see https://github.com/keras-team/tf-keras/issues/430
        callback_nanterminate,
        callback_plotting,
        callback_best_checkpoint,
    ]

    return callbacks
