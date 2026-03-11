import os
import numpy as np

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

from ml4fir.config import EXPERIMENTS_DIR, RESULTS_DIR

client = MlflowClient()


def append_and_save_csv(
    new_df, csv_path, index_col=None, dedup_col=None, index=False, json=False
):
    """
    Reads an existing CSV (if present), appends new_df, drops duplicates, and saves back to csv_path.

    Args:
        new_df (pd.DataFrame): The new data to append.
        csv_path (str): Path to the CSV file.
        index_col (str, optional): Column to set as index before saving.
        dedup_col (str, optional): Column to drop duplicates on.
        index (bool, optional): Whether to write row names (index). Default is False.
    """
    if os.path.exists(csv_path):
        old_df = pd.read_csv(csv_path)
        combined_df = pd.concat([old_df, new_df]).reset_index(drop=True)
        if dedup_col is not None:
            if not isinstance(dedup_col, list):
                dedup_col = [dedup_col]
            combined_df = combined_df.drop_duplicates(subset=dedup_col)
    else:
        combined_df = new_df.copy()
    if index_col:
        index = True
        combined_df.set_index(index_col, inplace=True)
    if not json:
        combined_df.to_csv(csv_path, index=index)
    else:
        combined_df.T.to_json(csv_path, orient="columns")


def save_df_with_lists_as_strings(
    df, csv_path, dedup_col=None, index=False, json=False
):
    """
    Converts all columns containing lists or dicts to strings, deduplicates, and saves to CSV.
    Args:
        df (pd.DataFrame): DataFrame to save.
        csv_path (str): Path to save CSV.
        dedup_col (str or list, optional): Column(s) to drop duplicates on.
        index (bool, optional): Whether to write row names (index). Default is False.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, (list, dict, np.ndarray))).any():
            df[col] = df[col].apply(str)
    append_and_save_csv(
        df, csv_path, dedup_col=dedup_col, index=index, json=json
    )


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
    base_results_path = RESULTS_DIR
    results_df = pd.concat(all_results).reset_index(drop=True)
    cross_validation_results_df = pd.concat(
        cross_validation_results_all
    ).reset_index(drop=True)
    grid_search_results_df = pd.concat(grid_search_results_all).reset_index(
        drop=True
    )
    back_projection_df_iso = pd.concat(back_projection_df_iso_all).reset_index(
        drop=True
    )

    for target_folder in targets_to_predict:
        experiment_target_folder = os.path.join(EXPERIMENTS_DIR, target_folder)
        os.makedirs(experiment_target_folder, exist_ok=True)
        experiment_target_file = os.path.join(
            experiment_target_folder, "experiment_configs.csv"
        )
        append_and_save_csv(
            configs_done_df,
            experiment_target_file,
            index_col="run_id",
            dedup_col="run_id",
        )

        target_results = results_df[
            results_df["target_variable"] == target_folder
        ]
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
            f"_{selected_group_fam}"
            if selected_group_fam
            else f"_{target_folder}"
        )

        target_results_path = os.path.join(
            final_results_path, f"results_summary{suffix_group}.csv"
        )
        save_df_with_lists_as_strings(
            target_results,
            target_results_path,
            dedup_col=target_results.columns.to_list(),
            index=False,
        )

        target_back_projection_iso = target_back_projection_iso.sort_values(
            "Wavenumber (cm⁻¹)", ascending=False
        ).reset_index(drop=True)
        target_back_projection_iso_path = os.path.join(
            final_results_path,
            f"results_summary{suffix_group}_back_projection.csv",
        )
        save_df_with_lists_as_strings(
            target_back_projection_iso,
            target_back_projection_iso_path,
            dedup_col=target_back_projection_iso.columns.to_list(),
            index=False,
        )

        target_grid_search_results_path = os.path.join(
            final_results_path,
            f"grid_search_results_{suffix_group}_back_projection.csv",
        )
        save_df_with_lists_as_strings(
            target_grid_search_results,
            target_grid_search_results_path,
            dedup_col=target_grid_search_results.columns.to_list(),
            index=False,
        )

        target_back_projection_iso["run_id"] = configs_done_df["run_id"].iloc[0]
        target_back_projection_iso["run_name"] = configs_done_df[
            "run_name"
        ].iloc[0]
        target_back_projection_iso_path = os.path.join(
            final_results_path,
            f"results_summary{suffix_group}_back_projection_best_per_experiment.csv",
        )
        if os.path.exists(target_back_projection_iso_path):
            old_back_projection = pd.read_csv(target_back_projection_iso_path)
            old_back_projection = old_back_projection[
                old_back_projection["run_id"]
                != target_back_projection_iso["run_id"].iloc[0]
            ]
            target_back_projection_iso = pd.concat(
                [old_back_projection, target_back_projection_iso]
            ).reset_index(drop=True)
            target_back_projection_iso.to_csv(
                target_back_projection_iso_path, index=False
            )
        else:
            target_back_projection_iso.to_csv(
                target_back_projection_iso_path, index=False
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
    # Identify best child run based on metric optimization direction
    if best_is_max:
        best_child = child_run_metrics_df[metric_to_choose].idxmax()
    else:
        best_child = child_run_metrics_df[metric_to_choose].idxmin()
    mlflow.log_metrics(
        child_run_metrics[best_child], run_id=mlflow_run_obj.info.run_id
    )
