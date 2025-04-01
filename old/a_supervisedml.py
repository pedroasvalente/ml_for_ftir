import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cross_decomposition import PLSRegression
from imblearn.over_sampling import SMOTE
from zz_config_module import random_seed
from zz_config_supervised_functions import random_forest, mlp_classifier, decision_tree, xboost
import os

# Define your group_fam of interest (or None to include all)
selected_group_fam = None  # Example: 'sedentary', 'ultrarunning', 'football'

# Load the cleaned dataset
file_path = "/Users/pedroasvalente/Library/CloudStorage/OneDrive-UniversidadedeCoimbra/02_WORK/05_DATA/002_phd_data_preprocess/001_3_cleaned_FTIR.csv"
df = pd.read_csv(file_path)

# Define target(s)
targets_to_predict = ["vo2max_classes_simplified"]
sample_types = df["sample_type"].unique()
ftir_start_col = "3050.854846120364"
ftir_columns = df.columns[df.columns.get_loc(ftir_start_col):]

# Prepare result containers
results = {
    'Sample Type': [],
    'Train Percentage': [],
    'Model': [],
    'Accuracy': [],
    'F1 Score': [],
    'ROC AUC': []
}

cross_validation_results = {
    'Sample Type': [],
    'Train Percentage': [],
    'Model': [],
    'Balanced Accuracy': [],
    'F1 Score': [],
    'Recall': [],
    'Precision': [],
    'Confusion Matrix': [],
    'Best Params': [],
    'mean_test_score': [],
    'std_test_score': [],
    'rank_test_score': [],
    'params': [],
    'best_index': [],
    'split0_test_score': [],
    'split1_test_score': [],
    'split2_test_score': [],
    'split3_test_score': [],
    'split4_test_score': [],
    'accuracy_score':[]
}

back_projection = {
    'Sample Type': [],
    'Train Percentage': [],
    'Model': [],
    'Accuracy': [],
    "Wavenumber (cm⁻¹)": [],
    "Importance": [],
}

# to use when being tested
# train_percentages = [0.8]


train_percentages = [0.8, 0.7, 0.6]

for target in targets_to_predict:
    print(f"\n>>> Starting Target: {target}\n")

    for sample_type in sample_types:
        print(f"\n--- Processing Sample Type: {sample_type} ---")
        sample_data = df[df["sample_type"] == sample_type].copy()

        if selected_group_fam:
            sample_data = sample_data[sample_data["group_fam"] == selected_group_fam]

        if sample_data[target].dropna().empty:
            print(f"[!] Skipping: No data for {target} in {sample_type}")
            continue

        spectral_data = sample_data[ftir_columns].astype(float)
        X_spectral_clean = spectral_data.loc[:, (spectral_data != 0).any(axis=0)]
        wavenumbers = X_spectral_clean.columns.values.astype(float)

        y = sample_data[target]
        valid_mask = y.notna()
        y = y[valid_mask]
        X = X_spectral_clean.loc[valid_mask]

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        for train_percentage in train_percentages:
            print(f"Training with {train_percentage * 100:.0f}% of the data")
            test_size = 1 - train_percentage

            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=random_seed, stratify=y_encoded
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            pls_da = PLSRegression(n_components=10)
            X_train_pls = pls_da.fit_transform(X_train_scaled, y_train)[0]
            X_test_pls = pls_da.transform(X_test_scaled)
            loadings = pls_da.x_weights_

            smote = SMOTE(random_state=random_seed)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_pls, y_train)
            #to use when being tested
            # for model_func in [random_forest]:
            for model_func in [random_forest, mlp_classifier, decision_tree, xboost]:
                results, cross_validation_results, back_projection = model_func(
                    X_train_resampled, y_train_resampled,
                    X_test_pls, y_test, label_encoder,
                    sample_type, train_percentage, loadings, wavenumbers,
                    results, cross_validation_results, target,
                    back_projection, group_fam_to_use=selected_group_fam)

# Create final results folder based on the target name
base_results_path = "000_final_results"
target_folder = targets_to_predict[0]  # You confirmed it's always one
final_results_path = os.path.join(base_results_path, target_folder)
os.makedirs(final_results_path, exist_ok=True)

# Define suffix for filenames (group_fam or just the target)
suffix_group = f"_{selected_group_fam}" if selected_group_fam else f"_{target_folder}"

# Save final results to Excel files in the correct path
pd.DataFrame(results).to_excel(os.path.join(final_results_path, f"results_summary{suffix_group}.xlsx"), index=False)
pd.DataFrame(cross_validation_results).to_excel(os.path.join(final_results_path, f"results_summary{suffix_group}_cross.xlsx"), index=False)
pd.DataFrame(back_projection).to_excel(os.path.join(final_results_path, f"results_summary{suffix_group}_back_projection.xlsx"), index=False)