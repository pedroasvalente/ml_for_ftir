import json
import os

os.environ["KERAS_BACKEND"] = "torch"

# Import config first — triggers dagshub.init() and sets MLflow tracking URI to DagsHub
from ml4fir.config import (
    EXPERIMENTS_DIR,
    PROCESSED_TRAINING_DATA_FILEPATH,
    logger,
    random_seed,
)

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from tqdm import tqdm

from ml4fir.data import DataHandler
from ml4fir.modeling.models import names_dict
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
    configurations_done = []

    with open(experiment_config) as config_file:
        config = json.load(config_file)

    selected_group_fam = config.get("selected_group_fam", None)
    searchs_hipermetrics = config.get("searchs_hipermetrics", [None])
    model_types_to_train = config.get("model_types_to_train", [None])
    train_percentages = config.get("train_percentages", [None])
    sample_types = config.get("sample_types", [None])
    targets_to_predict = config.get("targets_to_predict", [None])
    experiment_name = config.get("experiment_name", "FTIR Supervised Training")
    run_name = config.get("run_name", "demo")
    num_classes = config.get("num_classes", [None])
    timepoints = config.get("timepoints", [None])

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
        "num_classes": num_classes,
        "timepoints": timepoints,
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
            "num_classes": n_classes,
            "timepoints": tp,
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
        for n_classes in num_classes
        for tp in timepoints
    ]
    new_confs = pd.DataFrame(configurations)
    new_confs.loc[new_confs["num_classes"] == 1, "apply_smote_resampling"] = (
        False
    )
    new_confs.loc[new_confs["apply_pls"] == False, "n_components"] = None
    new_confs["timepoints"] = new_confs["timepoints"].apply(
        lambda x: str(x) if isinstance(x, list) else x
    )
    new_confs = new_confs.drop_duplicates()
    new_confs["timepoints"] = new_confs["timepoints"].apply(
        lambda x: json.loads(x.replace("'", "\"")) if isinstance(x, str) else x
    )
    configurations = new_confs.to_dict(orient="records")
    # NOTE: each experiment can only have one target!
    datahandler = DataHandler(
        data_path=PROCESSED_TRAINING_DATA_FILEPATH, target=targets_to_predict[0]
    )

    mlflow.set_experiment(experiment_name=experiment_name)
    run_name = f"{run_name}_{'_'.join(targets_to_predict)}"
    main_run_args = {
        "run_name": run_name,
        "nested": True,
    }

    done_mask = None
    # NOTE: Each experiment configuration supports only one target variable
    target_exp_res_path = os.path.join(
        EXPERIMENTS_DIR,
        targets_to_predict[0],
        "experiment_configs.csv",
    )
    if os.path.exists(target_exp_res_path):
        target_exp_res = pd.read_csv(target_exp_res_path)
        new_confs = pd.DataFrame(configurations)
        new_confs["model_type"] = new_confs["model_type"].map(names_dict)
        mask = new_confs["num_classes"] == 1
        new_confs.loc[mask, "model_type"] = (
            new_confs.loc[mask, "model_type"].astype(str) + " Regressor"
        )
        mask = new_confs["n_components"] != new_confs["n_components"]
        new_confs.loc[mask, "n_components"] = np.nan
        new_confs["search_to_use"] = new_confs["search_to_use"].map(
            {"grid": "GridSearchCV", "bayes": "BayesSearchCV"}
        )
        equal_columns = [
            f for f in new_confs.columns if f in target_exp_res.columns
        ]
        merged = new_confs[equal_columns].merge(
            target_exp_res[equal_columns].drop_duplicates(),
            how="left",
            indicator=True,
        )
        done_mask = merged["_merge"] == "both"
        done_mask = done_mask.to_numpy()
        # configurations=list(np.array(configurations)[~done_mask])
    if done_mask is not None:
        if done_mask.all():
            logger.info("All configurations already done.")
            return

    with mlflow.start_run(**main_run_args) as run:
        mlflow.log_artifact(experiment_config)
        # Process each configuration
        with tqdm(
            configurations, desc="Training Configurations"
        ) as progress_bar:
            for i, config in enumerate(progress_bar):
                if done_mask is not None:
                    if done_mask[i]:
                        print(f"Skipping configuration already done: {config}")
                        continue
                # Update the progress bar with the current configuration
                progress_bar.set_postfix(
                    target=config["target"],
                    search_to_use=config["search_to_use"],
                    model_type=config["model_type"],
                    sample_type=config["sample_type"],
                    train_percentage=config["train_percentage"],
                    apply_pls=config["apply_pls"],
                    n_components=config["n_components"],
                    scale=config["scale"],
                    apply_smote_resampling=config["apply_smote_resampling"],
                )

                target = config["target"]
                sample_type = config["sample_type"]
                train_percentage = config["train_percentage"]
                model_type = config["model_type"]
                search_to_use = config["search_to_use"]
                scale = config["scale"]
                apply_pls = config["apply_pls"]
                apply_smote_resampling = config["apply_smote_resampling"]
                n_components = config.get("n_components", None)
                if n_components != n_components:
                    n_components = None
                if n_components is not None:
                    n_components = int(n_components)
                n_classes = config["num_classes"]
                timepoint = config.get("timepoints", None)
                if n_classes == 1:
                    model_type = model_type.replace("_classifier", "")
                    model_type = f"{model_type}_regressor"
                    apply_smote_resampling = False

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
                search_run = [
                    f for f in child_runs if f.info.run_name == sample_type
                ]
                if len(search_run) > 0:
                    run_args["run_id"] = search_run[0].info.run_id

                with mlflow.start_run(**run_args) as sample_type_run:
                    mlflow.log_param("sample_type", config["sample_type"])
                    # Process sample data
                    datahandler.process_sample_data(
                        target=target,
                        sample_type=sample_type,
                        selected_group_fam=selected_group_fam,
                        num_classes=n_classes,
                        timepoint=timepoint,
                    )

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
                        main_run_id=run.info.run_id,
                    )

                    # Collect results
                    results = training_results["results"]
                    cross_validation_results = training_results[
                        "cross_validation_results"
                    ]
                    grid_search_results = training_results[
                        "grid_search_results"
                    ]
                    back_projection_df_iso = training_results[
                        "back_projection_df"
                    ]
                    configs_done = training_results["configs"]
                    configs_done["run_name"] = run_name
                    all_results.append(results)
                    cross_validation_results_all.append(
                        cross_validation_results
                    )
                    grid_search_results_all.append(grid_search_results)
                    back_projection_df_iso_all.append(back_projection_df_iso)
                    configurations_done.append(configs_done)

            if len(configurations_done) == 0:
                return
            # Save results
            save_results(
                targets_to_predict,
                all_results,
                cross_validation_results_all,
                grid_search_results_all,
                back_projection_df_iso_all,
                selected_group_fam,
                configurations_done,
            )
