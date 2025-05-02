import os

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from ml4fir.config import (
    confusion_matrix_plot_path,
    global_threshold,
    logger,
    roc_plot_path,
)

# TODO: Seaborn is not beaing used for anything?
sns.set(style="whitegrid")
matplotlib.use("Agg")


def get_plot_path(base_path, target_name, group_fam_to_use=None):
    """
    Generate and create a directory path for saving plots.

    Args:
        base_path (str): The base directory path.
        target_name (str): The name of the target variable.
        group_fam_to_use (str, optional): Additional grouping information to include in the folder name. Defaults to None.

    Returns
    -------
        str: The full path to the directory.
    """
    subfolder = (
        f"{target_name}_{group_fam_to_use}" if group_fam_to_use else f"{target_name}"
    )
    full_path = os.path.join(base_path, subfolder)
    os.makedirs(full_path, exist_ok=True)
    return full_path


def plot_confusion_matrix(
    y_test,
    y_pred,
    label_encoder,
    accuracy_score,
    sample_type,
    train_percentage,
    test_name,
    target_name,
    threshold=None,
    group_fam_to_use=None,
):
    """
    Plot and save a confusion matrix if the accuracy score meets the threshold.

    Args:
        y_test (array-like): True labels.
        y_pred (array-like): Predicted labels.
        label_encoder (LabelEncoder): Encoder for label classes.
        accuracy_score (float): Accuracy score of the model.
        sample_type (str): Type of sample (e.g., train/test).
        train_percentage (float): Percentage of data used for training.
        test_name (str): Name of the test.
        target_name (str): Name of the target variable.
        threshold (int, optional): Accuracy threshold for plotting. Defaults to global_threshold.
        group_fam_to_use (str, optional): Additional grouping information. Defaults to None.

    Returns
    -------
        None
    """
    if threshold is None:
        threshold = global_threshold
    if accuracy_score >= threshold / 100:
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(6, 5))

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            linewidths=0.5,
            linecolor='gray',
            cbar=False,
            xticklabels=label_encoder,
            yticklabels=label_encoder,
            ax=ax
        )

        group_info = f"_{group_fam_to_use}" if group_fam_to_use else ""
        ax.set_title(
            f"{target_name} - Confusion Matrix ({sample_type} | {train_percentage * 100:.0f}%)\n{test_name}{group_info}",
            pad=20,
        )

        plt.text(
            0.5,
            1.05,
            f"Accuracy: {accuracy_score * 100:.2f}%",
            fontsize=10,
            ha="center",
            transform=ax.transAxes,
        )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

        plt.tight_layout()
        # TODO: set pathas from project config
        save_path = get_plot_path(confusion_matrix_plot_path, target_name, group_fam_to_use)
        plot_filename = f"{target_name}_ConfMatrix_{sample_type}_{int(train_percentage * 100)}pct_{test_name}{group_info}.png"
        plot_filepath = os.path.join(save_path, plot_filename)

        plt.savefig(plot_filepath, dpi=300, bbox_inches="tight")
        mlflow.log_figure(fig, plot_filename)
        plt.close()
        logger.info(f"Seaborn Confusion Matrix saved as: {plot_filepath}")
    else:
        logger.info(
            f"[Skipped] Confusion matrix (Acc: {accuracy_score * 100:.2f}%) < threshold ({threshold}%)"
        )


def plot_roc_curve(
    y_test,
    y_prob,
    label_encoder,
    sample_type,
    train_percentage,
    test_accuracy,
    test_name,
    target_name,
    threshold=None,
    group_fam_to_use=None,
):
    """
    Plot and save a ROC curve if the accuracy score meets the threshold.

    Args:
        y_test (array-like): True labels.
        y_prob (array-like): Predicted probabilities for each class.
        label_encoder (LabelEncoder): Encoder for label classes.
        sample_type (str): Type of sample (e.g., train/test).
        train_percentage (float): Percentage of data used for training.
        test_accuracy (float): Accuracy score of the model.
        test_name (str): Name of the test.
        target_name (str): Name of the target variable.
        threshold (int, optional): Accuracy threshold for plotting. Defaults to global_threshold.
        group_fam_to_use (str, optional): Additional grouping information. Defaults to None.

    Returns
    -------
        float: ROC AUC score if plotted, otherwise 0.0.
    """
    if threshold is None:
        threshold = global_threshold
    if len(np.unique(y_test)) == len(label_encoder):
        roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
        if test_accuracy >= threshold / 100:
            logger.info(f"ROC AUC: {roc_auc:.4f}")
            # Create a figure object
            fig, ax = plt.subplots()

            for i in range(len(label_encoder)):
                fpr, tpr, _ = roc_curve(y_test, y_prob[:, i], pos_label=i)
                ax.plot(
                    fpr,
                    tpr,
                    label=f"Class {label_encoder[i]} (AUC={roc_auc:.2f})",
                )

            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            group_info = f"_{group_fam_to_use}" if group_fam_to_use else ""
            plt.title(
                f"{target_name} - ROC Curve ({sample_type} | {train_percentage*100:.0f}%)\n{test_name}{group_info}"
            )
            plt.legend(loc="best")

            # Dynamic path
            save_path = get_plot_path(roc_plot_path, target_name, group_fam_to_use)
            plot_filename = f"{target_name}_ROC_{sample_type}_{int(train_percentage*100)}pct_{test_name}{group_info}.png"
            plot_filepath = os.path.join(save_path, plot_filename)

            plt.savefig(plot_filepath, dpi=300, bbox_inches="tight")
            mlflow.log_figure(fig, plot_filename)
            plt.close()
            logger.info(f"ROC curve plot saved as: {plot_filepath}")
            return roc_auc
        logger.info(
            f"[Skipped] ROC curve (Acc: {test_accuracy * 100:.2f}%) < threshold ({threshold}%)"
        )
        return roc_auc
    logger.info("Skipping ROC curve due to missing classes.")
    return 0.0


def plot_wavenumber_importances(
    valid_wavenumbers,
    valid_importances,
    target_name,
    sample_type,
    train_percentage,
    test_name,
    group_suffix,
    save_path,
):
    """
    Generate a plot with a broken x-axis for wavenumber importances.

    Parameters
    ----------
        valid_wavenumbers (array-like): Array of valid wavenumbers.
        valid_importances (array-like): Array of corresponding importances.
        target_name (str): Name of the target variable.
        sample_type (str): Type of the sample.
        train_percentage (float): Training percentage (e.g., 0.8 for 80%).
        test_name (str): Identifier for the test/model scenario.
        group_suffix (str): Suffix for filenames (e.g., based on group family).
        save_path (str): Path to save the plot.

    Returns
    -------
        str: Path to the saved plot file.
    """
    # Mask the edges of the wavenumber range
    mask_left = valid_wavenumbers > 2500
    mask_right = valid_wavenumbers < 1850

    wav_left = valid_wavenumbers[mask_left]
    imp_left = valid_importances[mask_left]

    wav_right = valid_wavenumbers[mask_right]
    imp_right = valid_importances[mask_right]

    # Create the plot with a broken x-axis
    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(12, 6),
        sharey=True,
        gridspec_kw={"width_ratios": [1, 3], "wspace": 0.01},
        constrained_layout=True,
    )

    # Plot left and right sections
    ax_left.plot(wav_left, imp_left, color="b")
    ax_left.set_xlim(np.max(wav_left), np.min(wav_left))
    ax_left.set_ylim(0, np.max(valid_importances) * 1.05)
    ax_left.grid(True)

    ax_right.plot(wav_right, imp_right, color="b")
    ax_right.set_xlim(np.max(wav_right), np.min(wav_right))
    ax_right.set_ylim(0, np.max(valid_importances) * 1.05)
    ax_right.grid(True)

    # Invert x-axis for both sections
    ax_left.invert_xaxis()
    ax_right.invert_xaxis()

    # Add labels and title
    fig.text(0.04, 0.5, "Importance", va="center", rotation="vertical", fontsize=12)
    fig.text(0.5, 0.02, r"Wavenumber (cm$^{-1}$)", ha="center", fontsize=12)

    fig.suptitle(
        f"{target_name} - Principal Wavenumber Importances\n{sample_type}, {int(train_percentage * 100)}% Train - {test_name}{group_suffix}",
        fontsize=14,
    )

    # Save the plot
    plot_filename = f"{target_name}_principal_wavenumbers_{sample_type}_{int(train_percentage * 100)}pct_{test_name}{group_suffix}.png"
    plot_filepath = os.path.join(save_path, plot_filename)
    plt.savefig(plot_filepath, dpi=300, bbox_inches="tight")
    mlflow.log_figure(fig, plot_filename)
    plt.close()
    logger.info(f"Plot saved to: {plot_filepath}")

    return plot_filepath
