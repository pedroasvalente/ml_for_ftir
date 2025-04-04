# WHY? wtf is this? why is the file called config_features_importance.py?
# what are you trying to do here?

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ml4fir.config import global_threshold

sns.set(style="whitegrid")


def get_principal_wavenumber_path(target_name, group_fam_to_use=None):
    folder = f"000_principal_wavenumber/{target_name}"
    if group_fam_to_use:
        folder += f"_{group_fam_to_use}"
    os.makedirs(folder, exist_ok=True)
    return folder


def func_back_projection(
    lv_importances,
    pls_loadings,
    wavenumbers,
    test_accuracy,
    target_name,
    sample_type,
    train_percentage,
    test_name,
    threshold=global_threshold,
    group_fam_to_use=None,
):
    """
    Calculates and visualizes the contributions of wavenumbers for a predictive model.
    It identifies the most important wavenumbers based on their importances and plots their contributions,
    if the model's test accuracy meets or exceeds the specified threshold.
    Additionally, it excludes wavenumbers in the range 1850-2500 cm⁻¹ and presents a broken x-axis plot.

    Parameters:
      - lv_importances (numpy array): Importances of the model's latent variables.
      - pls_loadings (numpy array): PLS loadings used to calculate wavenumber contributions (transposed internally).
      - wavenumbers (numpy array): Array of wavenumbers corresponding to data points.
      - test_accuracy (float): Test set accuracy, used to decide whether to generate/save the plot.
      - target_name (str): Name of the target variable (e.g., "group_fam", "timepoint", etc.).
      - sample_type (str): Type of sample (e.g., "CAPILAR", "SERUM", etc.).
      - train_percentage (float): Fraction of data used for training (e.g., 0.8 for 80%).
      - test_name (str): Identifier for the test/model scenario (used in plot titles/filenames).
      - threshold (float): Minimum accuracy required to generate and save the plot (default: 60).
      - group_fam_to_use (str, optional): If provided, included in filenames and titles.

    Returns:
      - top_wavenumbers (numpy array): Array of the 20 most important wavenumbers (from the valid set).
      - top_importances (numpy array): Corresponding importance values.
    """
    print(f"[DEBUG] Threshold: {threshold}, Test Accuracy: {test_accuracy}")

    # Transpor os loadings
    pls_loadings = pls_loadings.transpose()
    wavenumber_importances = np.abs(lv_importances @ pls_loadings)
    wavenumber_importances /= wavenumber_importances.sum()

    # Remover zona da água
    valid_mask = (wavenumbers < 1850) | (wavenumbers > 2500)
    valid_wavenumbers = wavenumbers[valid_mask]
    valid_importances = wavenumber_importances[valid_mask]

    # Sufixo limpo
    group_suffix = f"_{group_fam_to_use}" if group_fam_to_use else ""
    save_path = get_principal_wavenumber_path(target_name, group_fam_to_use)

    # Guardar Excel
    if test_accuracy >= threshold / 100:
        df_out = pd.DataFrame(
            {"Wavenumber (cm⁻¹)": valid_wavenumbers, "Importance": valid_importances}
        )
        excel_filename = f"{target_name}_wavenumbers_importance_{sample_type}_{int(train_percentage * 100)}pct_{test_name}_accuracy_{test_accuracy:.4f}{group_suffix}.xlsx"
        excel_filepath = os.path.join(save_path, excel_filename)
        df_out.to_excel(excel_filepath, index=False)
        print(f"Excel file saved to: {excel_filepath}")

    # Top 20
    top_indices = np.argsort(valid_importances)[-20:][::-1]
    top_wavenumbers = valid_wavenumbers[top_indices]
    top_importances = valid_importances[top_indices]

    """ print("Top Contributing Wavenumbers:")
    for wn, imp in zip(top_wavenumbers, top_importances):
        print(f"Wavenumber: {wn}, Importance: {imp:.4f}")"""

    # Plot
    if test_accuracy >= threshold / 100:
        mask_left = valid_wavenumbers > 2500
        mask_right = valid_wavenumbers < 1850

        wav_left = valid_wavenumbers[mask_left]
        imp_left = valid_importances[mask_left]

        wav_right = valid_wavenumbers[mask_right]
        imp_right = valid_importances[mask_right]

        fig, (ax_left, ax_right) = plt.subplots(
            1,
            2,
            figsize=(12, 6),
            sharey=True,
            gridspec_kw={"width_ratios": [1, 3], "wspace": 0.01},
            constrained_layout=True,
        )

        ax_left.plot(wav_left, imp_left, color="b")
        ax_left.set_xlim(np.max(wav_left), np.min(wav_left))
        ax_left.set_ylim(0, np.max(valid_importances) * 1.05)
        ax_left.grid(True)

        ax_right.plot(wav_right, imp_right, color="b")
        ax_right.set_xlim(np.max(wav_right), np.min(wav_right))
        ax_right.set_ylim(0, np.max(valid_importances) * 1.05)
        ax_right.grid(True)

        ax_left.invert_xaxis()
        ax_right.invert_xaxis()

        fig.text(0.04, 0.5, "Importance", va="center", rotation="vertical", fontsize=12)
        fig.text(0.5, 0.02, r"Wavenumber (cm$^{-1}$)", ha="center", fontsize=12)

        fig.suptitle(
            f"{target_name} - Principal Wavenumber Importances\n{sample_type}, {int(train_percentage * 100)}% Train - {test_name}{group_suffix}",
            fontsize=14,
        )

        plot_filename = f"{target_name}_principal_wavenumbers_{sample_type}_{int(train_percentage * 100)}pct_{test_name}{group_suffix}.png"
        plot_filepath = os.path.join(save_path, plot_filename)
        plt.savefig(plot_filepath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to: {plot_filepath}")

    return top_wavenumbers, top_importances
