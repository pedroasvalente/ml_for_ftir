import os

import numpy as np
import pandas as pd

from ml4fir.config import EXPERIMENTS_DIR, TESTS_DIR
from ml4fir.ploting import (
    plot_confusion_matrix,
    plot_metrics_per_group,
    plot_roc_curve,
    plot_wavenumber_importances,
)


def test_plot_confusion_matrix():
    npz_path = os.path.join(
        TESTS_DIR, "ploting", "test_data", "plot_confusion_matrix.npz"
    )
    npz = np.load(npz_path, allow_pickle=True)
    npz_dict = dict(npz)
    for key in [
        "accuracy_score",
        "sample_type",
        "train_percentage",
        "test_name",
        "target_name",
        "threshold",
        "group_fam_to_use",
    ]:
        npz_dict[key] = npz_dict[key].item()
    npz_dict["accuracy_score"] = 0.75

    plot_confusion_matrix(
        plot_filepath=npz_path.replace(".npz", ".png"),
        mlflow_is_running=False,
        **npz_dict,
    )


def test_plot_roc_curve():
    npz_path = os.path.join(TESTS_DIR, "ploting", "test_data", "plot_roc_curve.npz")
    npz = np.load(npz_path, allow_pickle=True)
    npz_dict = dict(npz)
    for key in [
        "sample_type",
        "train_percentage",
        "test_name",
        "target_name",
        "threshold",
        "group_fam_to_use",
        "test_accuracy",
    ]:
        npz_dict[key] = npz_dict[key].item()
    npz_dict["test_accuracy"] = 0.75

    plot_roc_curve(
        plot_filepath=npz_path.replace(".npz", ".png"),
        mlflow_is_running=False,
        **npz_dict,
    )


def test_plot_wavenumber_importances():
    npz_path = os.path.join(
        TESTS_DIR, "ploting", "test_data", "plot_wavenumber_importances.npz"
    )
    npz = np.load(npz_path, allow_pickle=True)
    npz_dict = dict(npz)
    for key in [
        "sample_type",
        "train_percentage",
        "test_name",
        "target_name",
        "group_suffix",
    ]:
        npz_dict[key] = npz_dict[key].item()

    plot_wavenumber_importances(
        plot_filepath=npz_path.replace(".npz", ".png"),
        mlflow_is_running=False,
        **npz_dict,
    )


def test_plot_metrics_per_group():

    metric = "acc"
    target_name = "group_fam"
    experiment_file = os.path.join(
        EXPERIMENTS_DIR, target_name, "experiment_configs.csv"
    )
    metric_df = pd.read_csv(experiment_file)
    groupby = "sample_type"

    png_path = os.path.join(
        TESTS_DIR, "ploting", "test_data", "plot_metrics_per_group.png"
    )

    plot_metrics_per_group(
        metric_df,
        metric,
        groupby,
        target_name,
        mlflow_is_running=False,
        plot_filepath=png_path,
    )


test_plot_confusion_matrix()
test_plot_roc_curve()
test_plot_wavenumber_importances()
test_plot_metrics_per_group()
