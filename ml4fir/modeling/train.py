from loguru import logger
import pandas as pd
from tqdm import tqdm
import typer

from ml4fir.config import PROCESSED_TRAINING_DATA_FILEPATH, random_seed
from ml4fir.data.config import data_cols
from ml4fir.data.load_data import preprocess_data, process_sample_data
from ml4fir.modeling.train_utils import supervised_training

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    # features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    # model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):

    # Prepare result containers
    results = {
        "Sample Type": [],
        "Train Percentage": [],
        "Model": [],
        "Accuracy": [],
        "F1 Score": [],
        "ROC AUC": [],
    }

    cross_validation_results_all = []

    back_projection = {
        "Sample Type": [],
        "Train Percentage": [],
        "Model": [],
        "Accuracy": [],
        "Wavenumber (cm⁻¹)": [],
        "Importance": [],
    }

    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Training some model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Modeling training complete.")
    # -----------------------------------------

    df = pd.read_csv(PROCESSED_TRAINING_DATA_FILEPATH)

    # These seem like configurations
    targets_to_predict = ["group_fam"]
    train_percentages = [0.8, 0.7, 0.6]

    # This is to choose if you want to train on a specific group family or not
    # WHY? if this is your target?
    selected_group_fam = None

    # Sample types are trained separately.
    sample_types = df["sample_type"].unique()

    ftir_columns = df.columns[~df.columns.isin(data_cols)]

    for target in targets_to_predict:
        print(f"\n>>> Starting Target: {target}\n")

        for sample_type in sample_types:
            X, y_encoded, wavenumbers = process_sample_data(
                sample_data=df,
                target=target,
                sample_type=sample_type,
                ftir_columns=ftir_columns,
                selected_group_fam=selected_group_fam,
            )
            # Skip if no valid data
            if X is None or y_encoded is None:
                continue
            for train_percentage in train_percentages:
                # Preprocess the data
                X_train, X_test, y_train, y_test, loadings = preprocess_data(
                    X=X,
                    y_encoded=y_encoded,
                    train_percentage=train_percentage,
                    random_seed=random_seed,
                    scale=True,  # Enable scaling
                    apply_pls=True,  # Enable PLS-DA
                    apply_smote_resampling=True,  # Enable SMOTE
                    n_components=10,  # Number of PLS components
                )

                for model_type in ["random_forest", "mlp", "decision_tree", "xgboost"]:
                    results, cross_validation_results, back_projection = (
                        supervised_training(
                            x_train=X_train,
                            y_train=y_train,
                            x_test=X_test,
                            y_test=y_test,
                            label_encoder=None,
                            sample_type=sample_type,
                            train_percentage=train_percentage,
                            loadings=loadings,
                            wavenumbers=wavenumbers,
                            results=results,
                            cross_validation_results=cross_validation_results,
                            target_column=target,
                            back_projection=back_projection,
                            model_type=model_type,
                            group_fam_to_use=selected_group_fam,
                        )
                    )
                    cross_validation_results_all.append(cross_validation_results)

    # Create final results folder based on the target name
    base_results_path = "000_final_results"
    target_folder = targets_to_predict[0]  # You confirmed it's always one
    final_results_path = os.path.join(base_results_path, target_folder)
    os.makedirs(final_results_path, exist_ok=True)

    # Define suffix for filenames (group_fam or just the target)
    suffix_group = (
        f"_{selected_group_fam}" if selected_group_fam else f"_{target_folder}"
    )

    # Save final results to Excel files in the correct path
    pd.DataFrame(results).to_excel(
        os.path.join(final_results_path, f"results_summary{suffix_group}.xlsx"),
        index=False,
    )
    pd.DataFrame(cross_validation_results).to_excel(
        os.path.join(final_results_path, f"results_summary{suffix_group}_cross.xlsx"),
        index=False,
    )
    pd.DataFrame(back_projection).to_excel(
        os.path.join(
            final_results_path, f"results_summary{suffix_group}_back_projection.xlsx"
        ),
        index=False,
    )


# TODO: check ruff errors, they are the lead to next things to solve.
# Dicts are being changed inside functions, too many things are passed into this supervised learning function., probably best to slip it further.abs
# TODO: make this loop not a loop but entrie points for tqdm
# TODO: isolate each step..abs
# TODO: add mlflow tracking

if __name__ == "__main__":
    app()
