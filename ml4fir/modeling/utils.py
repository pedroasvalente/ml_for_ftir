import os

import pandas as pd

from ml4fir.config import RESULTS_DIR


def save_results(
    targets_to_predict,
    all_results,
    cross_validation_results_all,
    grid_search_results_all,
    back_projection_df_iso_all,
    selected_group_fam,
):
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
