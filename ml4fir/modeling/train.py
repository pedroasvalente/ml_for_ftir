import json

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
from tqdm import tqdm

# logging.getLogger("mlflow").setLevel(logging.DEBUG)
mlflow.autolog(log_datasets=False)


from ml4fir.config import PROCESSED_TRAINING_DATA_FILEPATH, logger, random_seed
from ml4fir.data import DataHandler
from ml4fir.modeling.train_utils import supervised_training
from ml4fir.modeling.utils import save_results

client = MlflowClient()


def train(
    experiment_config: str = None,
):

    # Prepare result containers
    all_results = []
    cross_validation_results_all = []
    back_projection_all = []
    grid_search_results_all = []
    back_projection_df_iso_all = []

    datahandler = DataHandler(data_path=PROCESSED_TRAINING_DATA_FILEPATH)

    with open(experiment_config) as config_file:
        config = json.load(config_file)

    selected_group_fam = config.get("selected_group_fam", None)
    searchs_hipermetrics = config.get("searchs_hipermetrics", [])
    model_types_to_train = config.get("model_types_to_train", [])
    train_percentages = config.get("train_percentages", [])
    sample_types = config.get("sample_types", [])
    targets_to_predict = config.get("targets_to_predict", [])
    experiment_name = config.get("experiment_name", "FTIR Supervised Training")
    run_name = config.get("run_name", "demo")

    scale_normalization = config.get("scale", [True])
    PLS_regression = config.get("apply_pls", [True])
    smote_resampling = config.get("apply_smote_resampling", [True])
    n_components_list = config.get("n_components", [10])
    if not np.any(PLS_regression):
        n_components_list = [None]

    configurations_dict = {
        "search_to_use": searchs_hipermetrics,
        "model_type": model_types_to_train,
        "train_percentage": train_percentages,
        "sample_type": sample_types,
        "target": targets_to_predict,
        "scale": scale_normalization,
        "apply_pls": PLS_regression,
        "apply_smote_resampling": smote_resampling,
        "n_components": n_components_list,
    }

    # Create a list of configurations
    configurations = [
        {
            "search_to_use": search_to_use,
            "model_type": model_type,
            "train_percentage": train_percentage,
            "sample_type": sample_type,
            "target": target,
            "scale": scale,
            "apply_pls": apply_pls,
            "apply_smote_resampling": apply_smote_resampling,
            "n_components": n_c,
        }
        for search_to_use in searchs_hipermetrics
        for model_type in model_types_to_train
        for train_percentage in train_percentages
        for sample_type in sample_types
        for target in targets_to_predict
        for scale in scale_normalization
        for apply_pls in PLS_regression
        for apply_smote_resampling in smote_resampling
        for n_c in n_components_list
    ]

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name, nested=True) as run:
        mlflow.log_params(configurations_dict)

        # Process each configuration
        with tqdm(configurations, desc="Training Configurations") as progress_bar:
            for config in progress_bar:
                # Update the progress bar with the current configuration
                progress_bar.set_postfix(
                    search_to_use=config["search_to_use"],
                    model_type=config["model_type"],
                    train_percentage=config["train_percentage"],
                    sample_type=config["sample_type"],
                    target=config["target"],
                )

                target = config["target"]
                sample_type = config["sample_type"]
                train_percentage = config["train_percentage"]
                model_type = config["model_type"]
                search_to_use = config["search_to_use"]
                scale = config["scale"]
                apply_pls = config["apply_pls"]
                apply_smote_resampling = config["apply_smote_resampling"]
                n_components = config["n_components"]

                logger.info(f">>> Starting Target: {target}")

                run_args = {
                    "run_name": f"{sample_type}",
                    "nested": True,
                    "parent_run_id": run.info.run_id,
                }

                # Search for child runs using the parent run ID
                child_runs = client.search_runs(
                    experiment_ids=[run.info.experiment_id],
                    filter_string=f"tags.mlflow.parentRunId = '{run.info.run_id}'",
                )
                search_run = [f for f in child_runs if f.info.run_name == sample_type]
                if len(search_run) > 0:
                    run_args["run_id"] = search_run[0].info.run_id

                with mlflow.start_run(**run_args) as sample_type_run:
                    mlflow.log_param("sample_type", config["sample_type"])
                    # Process sample data
                    datahandler.process_sample_data(
                        target=target,
                        sample_type=sample_type,
                        selected_group_fam=selected_group_fam,
                    )
                    # dataset = datahandler.get_mlflow_dataset_complete()
                    # mlflow.log_input(
                    #     dataset,
                    #     context="Complete",
                    #     tags={
                    #         "target": target,
                    #         "sample_type": sample_type,
                    #     },
                    # )

                    # Skip if no valid data
                    if datahandler.X is None or datahandler.y_encoded is None:
                        logger.warning(
                            f"Skipping configuration due to invalid data: {config}"
                        )
                        continue

                    # Preprocess the data
                    mlflow.autolog(disable=True)
                    datahandler.preprocess_data(
                        train_percentage=train_percentage,
                        random_seed=random_seed,
                        scale=scale,
                        apply_pls=apply_pls,
                        apply_smote_resampling=apply_smote_resampling,
                        n_components=n_components,
                    )
                    mlflow.autolog(log_datasets=False)

                    # dataset_train, dataset_test = datahandler.get_mlflow_dataset()
                    # tags = {
                    #     "parent_dataset": dataset.name,
                    #     "random_seed": random_seed,
                    #     "scale": scale,
                    #     "apply_pls": apply_pls,
                    #     "apply_smote_resampling": apply_smote_resampling,
                    #     "n_components": n_components,
                    #     "train_percentage": train_percentage,
                    # }
                    # tags = {k: str(v) for k, v in tags.items()}
                    # mlflow.log_input(dataset_train, context="Train", tags=tags)
                    # mlflow.log_input(dataset_test, context="Eval", tags=tags)

                    # Train the model
                    training_results = supervised_training(
                        datahandler=datahandler,
                        sample_type=sample_type,
                        train_percentage=train_percentage,
                        target_column=target,
                        model_type=model_type,
                        group_fam_to_use=selected_group_fam,
                        mlflow_run=sample_type_run,
                        search_to_use=search_to_use,
                    )

                    # Collect results
                    results = training_results["results"]
                    cross_validation_results = training_results[
                        "cross_validation_results"
                    ]
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


# TODO: only train the model once, and save the focker, probably done with mlflow implement it 1st
