import os

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

from ml4fir.config import EXPERIMENTS_DIR, RESULTS_DIR

client = MlflowClient()


def save_results(
    targets_to_predict,
    all_results,
    cross_validation_results_all,
    grid_search_results_all,
    back_projection_df_iso_all,
    selected_group_fam,
    configs_done,
):

    configs_done_df = pd.concat(configs_done)
    # mlflow.log_table(configs_done_df, "experiment_configs_done.json")
    configs_done_df = configs_done_df.set_index("run_id")

    base_results_path = RESULTS_DIR
    results_df = pd.concat(all_results).reset_index(drop=True)
    cross_validation_results_df = pd.concat(cross_validation_results_all).reset_index(
        drop=True
    )
    grid_search_results_df = pd.concat(grid_search_results_all).reset_index(drop=True)
    back_projection_df_iso = pd.concat(back_projection_df_iso_all).reset_index(
        drop=True
    )

    for target_folder in targets_to_predict:

        experiment_target_folder = os.path.join(EXPERIMENTS_DIR, target_folder)
        os.makedirs(experiment_target_folder, exist_ok=True)
        experiment_target_file = os.path.join(
            experiment_target_folder, "experiment_configs.csv"
        )
        if os.path.exists(experiment_target_file):
            configs_done_df_old = pd.read_csv(experiment_target_file)
            configs_done_df = pd.concat([configs_done_df_old, configs_done])
            configs_done_df = configs_done_df.drop_duplicates(subset=["run_id"])
        configs_done_df.to_csv(experiment_target_file)

        target_results = results_df[results_df["target_variable"] == target_folder]
        target_cross_validation_results = cross_validation_results_df[
            cross_validation_results_df["target_variable"] == target_folder
        ]
        target_grid_search_results = grid_search_results_df[
            grid_search_results_df["target_variable"] == target_folder
        ]
        target_back_projection_iso = back_projection_df_iso[
            back_projection_df_iso["target_variable"] == target_folder
        ]

        final_results_path = os.path.join(base_results_path, target_folder)
        os.makedirs(final_results_path, exist_ok=True)

        suffix_group = (
            f"_{selected_group_fam}" if selected_group_fam else f"_{target_folder}"
        )

        # Save results to CSV
        target_results.to_csv(
            os.path.join(final_results_path, f"results_summary{suffix_group}.csv"),
            index=False,
        )

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

        # Save results to Excel
        # target_cross_validation_results.to_excel(
        #     os.path.join(
        #         final_results_path, f"results_summary{suffix_group}_cross.xlsx"
        #     ),
        #     index=False,
        # )
        # TODO: probably get this were in another way.
        # TODO: get some sort of ID. mlflow?
        # Save results to JSON
        target_cross_validation_results.T.to_json(
            os.path.join(
                final_results_path, f"results_summary{suffix_group}_cross.json"
            ),
            orient="columns",
        )


def log_best_child(
    mlflow_run_obj, metric_to_choose="acc", best_is_max=True, save_model=False
):
    child_runs = client.search_runs(
        experiment_ids=[mlflow_run_obj.info.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{mlflow_run_obj.info.run_id}'",
    )

    # Collect metrics from child runs
    child_run_metrics = {}
    for child_run in child_runs:
        run_id = child_run.info.run_id
        metrics = child_run.data.metrics  # Get metrics from the child run
        child_run_metrics[run_id] = metrics
    child_run_metrics_df = pd.DataFrame(child_run_metrics).T
    # TODO: probably make a metrics handler
    if best_is_max:
        best_child = child_run_metrics_df[metric_to_choose].idxmax()
    else:
        best_child = child_run_metrics_df[metric_to_choose].idxmin()
    mlflow.log_metrics(child_run_metrics[best_child], run_id=mlflow_run_obj.info.run_id)
    # if save_model:
    #     best_child_run = client.get_run(best_child)
    #     [f for f in mlflow.artifacts.list_artifacts(run_id=best_child) if f.path=="best model"][0]
