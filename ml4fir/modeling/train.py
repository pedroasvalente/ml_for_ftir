from loguru import logger
import mlflow
import pandas as pd
from tqdm import tqdm
import typer

from ml4fir.config import PROCESSED_TRAINING_DATA_FILEPATH, random_seed
from ml4fir.data import DataHandler
from ml4fir.data.config import data_cols
from ml4fir.modeling.train_utils import supervised_training
from ml4fir.modeling.utils import save_results

mlflow.autolog()
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

    df = pd.read_csv(PROCESSED_TRAINING_DATA_FILEPATH)
    datahandler = DataHandler(data_path=PROCESSED_TRAINING_DATA_FILEPATH)

    # Define configurations
    targets_to_predict = ["group_fam"]
    train_percentages = [0.8, 0.7, 0.6]
    train_percentages = [0.8]

    sample_types = df["sample_type"].unique()
    sample_types = ["CAPILAR"]
    model_types_to_train = [
        "random_forest",
        "mlp_classifier",
        "decision_tree",
        "xgboost",
    ]
    selected_group_fam = None
    ftir_columns = df.columns[~df.columns.isin(data_cols)]

    # Create a list of configurations
    configurations = [
        {
            "target": target,
            "sample_type": sample_type,
            "train_percentage": train_percentage,
            "model_type": model_type,
        }
        for target in targets_to_predict
        for sample_type in sample_types
        for train_percentage in train_percentages
        for model_type in model_types_to_train
    ]
    mlflow.set_experiment("FTIR Supervised Training - Phase 1")
    with mlflow.start_run(run_name="demo") as run:

        # Process each configuration
        with tqdm(configurations, desc="Training Configurations") as progress_bar:
            for config in progress_bar:
                # Update the progress bar with the current configuration
                progress_bar.set_postfix(
                    target=config["target"],
                    sample_type=config["sample_type"],
                    train_percentage=config["train_percentage"],
                    model_type=config["model_type"],
                )

                target = config["target"]
                sample_type = config["sample_type"]
                train_percentage = config["train_percentage"]
                model_type = config["model_type"]
                logger.info(f"\n>>> Starting Target: {target}\n")

                # Process sample data
                datahandler.process_sample_data(
                    target=target,
                    sample_type=sample_type,
                    ftir_columns=ftir_columns,
                    selected_group_fam=selected_group_fam,
                )
                dataset = datahandler.get_mlflow_dataset_complete()
                mlflow.log_input(
                    dataset,
                    context="Complete",
                    tags={
                        "target": target,
                        "sample_type": sample_type,
                    },
                )

                # Skip if no valid data
                if datahandler.X is None or datahandler.y_encoded is None:
                    logger.warning(
                        f"Skipping configuration due to invalid data: {config}"
                    )
                    continue

                # Preprocess the data
                scale = True
                apply_pls = True
                apply_smote_resampling = True
                n_components = 10
                datahandler.preprocess_data(
                    train_percentage=train_percentage,
                    random_seed=random_seed,
                    scale=scale,
                    apply_pls=apply_pls,
                    apply_smote_resampling=apply_smote_resampling,
                    n_components=n_components,
                )
                dataset_train, dataset_test = datahandler.get_mlflow_dataset()
                tags = {
                    "parent_dataset": dataset.name,
                    "random_seed": random_seed,
                    "scale": scale,
                    "apply_pls": apply_pls,
                    "apply_smote_resampling": apply_smote_resampling,
                    "n_components": n_components,
                    "train_percentage": train_percentage,
                }
                tags = {k: str(v) for k, v in tags.items()}
                mlflow.log_input(dataset_train, context="Train", tags=tags)
                mlflow.log_input(dataset_test, context="Eval", tags=tags)

                # Train the model
                training_results = supervised_training(
                    datahandler=datahandler,
                    sample_type=sample_type,
                    train_percentage=train_percentage,
                    target_column=target,
                    model_type=model_type,
                    group_fam_to_use=selected_group_fam,
                )

                # Collect results
                results = training_results["results"]
                cross_validation_results = training_results["cross_validation_results"]
                grid_search_results = training_results["grid_search_results"]
                back_projection_df_iso = training_results["back_projection_df"]

                all_results.append(results)
                cross_validation_results_all.append(cross_validation_results)
                grid_search_results_all.append(grid_search_results)
                back_projection_df_iso_all.append(back_projection_df_iso)

            # Save results
            save_results(
                targets_to_predict,
                all_results,
                cross_validation_results_all,
                grid_search_results_all,
                back_projection_df_iso_all,
                selected_group_fam,
            )


# Change the logging stuff, like the line where the log is
# TODO: isolate each step..abs
# TODO: add mlflow tracking
# TODO: only train the model once, and save the focker, probably done with mlflow implement it 1st
if __name__ == "__main__":
    app()
