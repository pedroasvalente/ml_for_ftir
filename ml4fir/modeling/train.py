import os

from loguru import logger
import pandas as pd
from tqdm import tqdm
import typer

from ml4fir.config import PROCESSED_TRAINING_DATA_FILEPATH, random_seed
from ml4fir.data import DataHandler
from ml4fir.data.config import data_cols
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
    all_results = []
    cross_validation_results_all = []
    back_projection_all = []
    grid_search_results_all = []
    back_projection_df_iso_all = []

    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Training some model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Modeling training complete.")
    # -----------------------------------------

    df = pd.read_csv(PROCESSED_TRAINING_DATA_FILEPATH)
    datahandler = DataHandler(data_path=PROCESSED_TRAINING_DATA_FILEPATH)
    # These seem like configurations
    targets_to_predict = ["group_fam"]
    train_percentages = [0.8, 0.7, 0.6]

    # This is to choose if you want to train on a specific group family or not
    # WHY? if this is your target?
    selected_group_fam = None

    # Sample types are trained separately.
    sample_types = df["sample_type"].unique()

    # model types to train
    model_types_to_train = [
        "random_forest",
        "mlp_classifier",
        "decision_tree",
        "xgboost",
    ]

    ftir_columns = df.columns[~df.columns.isin(data_cols)]
    for target in targets_to_predict:
        print(f"\n>>> Starting Target: {target}\n")

        for sample_type in sample_types:
            datahandler.process_sample_data(
                target=target,
                sample_type=sample_type,
                ftir_columns=ftir_columns,
                selected_group_fam=selected_group_fam,
            )
            # Skip if no valid data
            if datahandler.X is None or datahandler.y_encoded is None:
                continue
            for train_percentage in train_percentages:
                # Preprocess the data
                datahandler.preprocess_data(
                    train_percentage=train_percentage,
                    random_seed=random_seed,
                    scale=True,
                    apply_pls=True,
                    apply_smote_resampling=True,
                    n_components=10,
                )

                for model_type in model_types_to_train:
                    training_results = supervised_training(
                        datahandler=datahandler,
                        sample_type=sample_type,
                        train_percentage=train_percentage,
                        target_column=target,
                        model_type=model_type,
                        group_fam_to_use=selected_group_fam,
                    )

                    results = training_results["results"]
                    cross_validation_results = training_results[
                        "cross_validation_results"
                    ]
                    back_projection = training_results["back_projection"]
                    grid_search_results = training_results["grid_search_results"]
                    back_projection_df_iso = training_results["back_projection_df"]

                    cross_validation_results_all.append(cross_validation_results)
                    all_results.append(results)
                    back_projection_all.append(back_projection)
                    grid_search_results_all.append(grid_search_results)
                    back_projection_df_iso_all.append(back_projection_df_iso)
    # Create final results folder based on the target name
    base_results_path = "000_final_results"
    target_folder = targets_to_predict[0]  # You confirmed it's always one
    results_df = pd.concat(all_results)
    cross_validation_results_df = pd.concat(cross_validation_results_all)
    back_projection_df = pd.concat(back_projection_all)
    grid_search_results_df = pd.concat(grid_search_results_all)
    back_projection_df_iso = pd.concat(back_projection_df_iso_all)
    for target_folder in targets_to_predict:
        target_results = results_df[results_df["target_variable"] == target_folder]
        target_cross_validation_results = cross_validation_results_df[
            cross_validation_results_df["target_variable"] == target_folder
        ]
        target_back_projection = back_projection_df[
            back_projection_df["target_variable"] == target_folder
        ]
        target_grid_search_results = grid_search_results_df[
            grid_search_results_df["target_variable"] == target_folder
        ]
        target_back_projection_iso = back_projection_df_iso[
            back_projection_df_iso["target_variable"] == target_folder
        ]
        # TODO: make sure the df are only with the target name
        final_results_path = os.path.join(base_results_path, target_folder)
        os.makedirs(final_results_path, exist_ok=True)

        # Define suffix for filenames (group_fam or just the target)
        suffix_group = (
            f"_{selected_group_fam}" if selected_group_fam else f"_{target_folder}"
        )

        # Save final results to csv files in the correct path
        target_results.to_csv(
            os.path.join(final_results_path, f"results_summary{suffix_group}.csv"),
            index=False,
        )
        target_cross_validation_results.to_csv(
            os.path.join(
                final_results_path, f"results_summary{suffix_group}_cross.csv"
            ),
            index=False,
        )
        # WHY: is this csv better than the excel? does this have what you need?
        target_back_projection_iso.to_csv(
            os.path.join(
                final_results_path, f"results_summary{suffix_group}_back_projection.csv"
            ),
            index=False,
        )
        target_grid_search_results.to_csv(
            os.path.join(
                final_results_path,
                f"grid_search_results_{suffix_group}_back_projection.csv",
            ),
            index=False,
        )

        # Save final results to Excel files in the correct path
        target_cross_validation_results.to_excel(
            os.path.join(
                final_results_path, f"results_summary{suffix_group}_cross.xlsx"
            ),
            index=False,
        )
        target_back_projection.to_excel(
            os.path.join(
                final_results_path,
                f"results_summary{suffix_group}_back_projection.xlsx",
            ),
            index=False,
        )


# TODO: make this loop not a loop but entrie points for tqdm
# TODO: isolate each step..abs
# TODO: add mlflow tracking
# TODO: only train the model once, and save the focker, probably done with mlflow implement it 1st
if __name__ == "__main__":
    app()
