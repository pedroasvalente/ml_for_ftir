from loguru import logger
import pandas as pd
from tqdm import tqdm
import typer

from ml4fir.config import PROCESSED_TRAINING_DATA_FILEPATH, random_seed
from ml4fir.data import DataHandler
from ml4fir.data.config import data_cols
from ml4fir.modeling.train_utils import supervised_training
from ml4fir.modeling.utils import save_results

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

    # # ---- REPLACE THIS WITH YOUR OWN CODE ----
    # logger.info("Training some model...")
    # for i in tqdm(range(10), total=10):
    #     if i == 5:
    #         logger.info("Something happened for iteration 5.")
    # logger.success("Modeling training complete.")
    # # -----------------------------------------

    df = pd.read_csv(PROCESSED_TRAINING_DATA_FILEPATH)
    datahandler = DataHandler(data_path=PROCESSED_TRAINING_DATA_FILEPATH)

    # Define configurations
    targets_to_predict = ["group_fam"]
    train_percentages = [0.8, 0.7, 0.6]
    sample_types = df["sample_type"].unique()
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

    # Process each configuration
    for config in tqdm(configurations, desc="Training Configurations"):
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

        # Skip if no valid data
        if datahandler.X is None or datahandler.y_encoded is None:
            logger.warning(f"Skipping configuration due to invalid data: {config}")
            continue

        # Preprocess the data
        datahandler.preprocess_data(
            train_percentage=train_percentage,
            random_seed=random_seed,
            scale=True,
            apply_pls=True,
            apply_smote_resampling=True,
            n_components=10,
        )

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
