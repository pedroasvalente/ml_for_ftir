import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

global_threshold = 70


def get_plot_path(base_path, target_name, group_fam_to_use=None):
    """
    Generate and create a directory path for saving plots.

    Args:
        base_path (str): The base directory path.
        target_name (str): The name of the target variable.
        group_fam_to_use (str, optional): Additional grouping information to include in the folder name. Defaults to None.

    Returns:
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

    Returns:
        None
    """
    if threshold is None:
        threshold = global_threshold
    if accuracy_score >= threshold / 100:
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=label_encoder.classes_
        )

        fig, ax = plt.subplots()
        disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(False)
        ax.tick_params(length=0)

        group_info = f"_{group_fam_to_use}" if group_fam_to_use else ""

        ax.set_title(
            f"{target_name} - Confusion Matrix ({sample_type} | {train_percentage*100:.0f}%)\n{test_name}{group_info}",
            pad=20,
        )
        plt.text(
            0.5,
            1.02,
            f"Accuracy: {accuracy_score * 100:.2f}%",
            fontsize=10,
            ha="center",
            transform=ax.transAxes,
        )

        # Dynamic path
        save_path = get_plot_path("000_CM_plots", target_name, group_fam_to_use)
        plot_filename = f"{target_name}_ConfMatrix_{sample_type}_{int(train_percentage*100)}pct_{test_name}{group_info}.png"
        plot_filepath = os.path.join(save_path, plot_filename)

        plt.savefig(plot_filepath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Confusion Matrix plot saved as: {plot_filepath}")
    else:
        print(
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

    Returns:
        float: ROC AUC score if plotted, otherwise 0.0.
    """
    if threshold is None:
        threshold = global_threshold
    if len(np.unique(y_test)) == len(label_encoder.classes_):
        roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
        if test_accuracy >= threshold / 100:
            print(f"ROC AUC: {roc_auc:.4f}")
            for i in range(len(label_encoder.classes_)):
                fpr, tpr, _ = roc_curve(y_test, y_prob[:, i], pos_label=i)
                plt.plot(
                    fpr,
                    tpr,
                    label=f"Class {label_encoder.classes_[i]} (AUC={roc_auc:.2f})",
                )

            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            group_info = f"_{group_fam_to_use}" if group_fam_to_use else ""
            plt.title(
                f"{target_name} - ROC Curve ({sample_type} | {train_percentage*100:.0f}%)\n{test_name}{group_info}"
            )
            plt.legend(loc="best")

            # Dynamic path
            save_path = get_plot_path("000_ROC_plots", target_name, group_fam_to_use)
            plot_filename = f"{target_name}_ROC_{sample_type}_{int(train_percentage*100)}pct_{test_name}{group_info}.png"
            plot_filepath = os.path.join(save_path, plot_filename)

            plt.savefig(plot_filepath, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"ROC curve plot saved as: {plot_filepath}")
            return roc_auc
        else:
            print(
                f"[Skipped] ROC curve (Acc: {test_accuracy * 100:.2f}%) < threshold ({threshold}%)"
            )
            return roc_auc
    else:
        print("Skipping ROC curve due to missing classes.")
        return 0.0
