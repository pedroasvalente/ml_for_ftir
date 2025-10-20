import os
import shutil

import mlflow
import numpy as np
import pandas as pd

from ml4fir.config import (
    EXPERIMENTS_DIR,
    FIGURES_DIR,
    MLFLOW_ARTIFACTS_DIR,
    MODELS_DIR,
    PROCESSED_TRAINING_DATA_FILEPATH,
    PROJ_ROOT,
    logger,
    main_metric,
)
from ml4fir.data import DataHandler
from ml4fir.modeling.train_utils import evaluate_model
from ml4fir.ploting.ploting_functions import (
    plot_confusion_matrix,
    plot_metrics_per_group,
    plot_roc_curve,
    plot_wavenumber_importances,
)


def plot(
    target_to_predict: str,
    plots_to_do: list,
    only_best: bool = True,
    metric: str = main_metric,
    best_is_max: bool = True,
    sample_type: str = None,
):
    experiment_results_file = os.path.join(
        EXPERIMENTS_DIR, target_to_predict, "experiment_configs.csv"
    )
    experiment_results = pd.read_csv(experiment_results_file, index_col=0)
    if sample_type is not None:
        experiment_results = experiment_results[
            experiment_results["sample_type"] == sample_type
        ]

    # Select best experiment based on metric
    if best_is_max:
        best_experiment = experiment_results[metric].idxmax()
    else:
        best_experiment = experiment_results[metric].idxmin()
    row_best_experiment = experiment_results.loc[best_experiment]

    if only_best:
        all_best_experiment = row_best_experiment
    else:
        all_best_experiment = experiment_results

    for i in range(len(all_best_experiment)):
        row_best_experiment = all_best_experiment.iloc[i]
        best_experiment = row_best_experiment.name
        columns_for_data = [
            "scale",
            "apply_pls",
            "apply_smote_resampling",
            "n_components",
        ]
        data_args = row_best_experiment[columns_for_data].to_dict()
        train_percentage = row_best_experiment["train_percentage"]

        datahandler = DataHandler(
            data_path=PROCESSED_TRAINING_DATA_FILEPATH,
            target=target_to_predict,
            train=False,
            **data_args,
        )
        datahandler.process_sample_data(
            target=target_to_predict,
            sample_type=row_best_experiment["sample_type"],
        )

        datahandler.preprocess_data(train_percentage=train_percentage)
        x_train = datahandler.x_train
        x_test = datahandler.x_test
        y_test = datahandler.y_test
        label_encoder = datahandler.labels
        loadings = datahandler.loadings
        wavenumbers = datahandler.wavenumbers

        best_model_path = os.path.join(
            MODELS_DIR, target_to_predict, "best_model"
        )
        if sample_type is not None:
            best_model_path = os.path.join(
                MODELS_DIR, target_to_predict, sample_type, "best_model"
            )
        if not only_best:
            mlflow_best_model_path = os.path.join(
                MLFLOW_ARTIFACTS_DIR,
                str(row_best_experiment["experiment_id"]),
                best_experiment,
                "artifacts",
                "best model",
            )
            best_model_path = mlflow_best_model_path

        trained_model = mlflow.pyfunc.load_model(best_model_path)
        predictions = trained_model.predict(x_test)
        y_prob = trained_model.get_raw_model().predict_proba(x_test)

        test_name = f"{row_best_experiment['model_type']} ({row_best_experiment['search_to_use']})"

        do_principal = True
        try:
            y_pred, y_prob, metrics, lv_importance = evaluate_model(
                trained_model.get_raw_model(),
                x_test,
                y_test,
                x_train,
                row_best_experiment["model_type"],
            )
        except Exception as e:
            logger.info(f"Error evaluating model: {e}")
            do_principal = False

        for plot in plots_to_do:
            plot_folder = os.path.join(FIGURES_DIR, plot, target_to_predict)
            if only_best:
                plot_folder = os.path.join(plot_folder, "best_model")
            os.makedirs(plot_folder, exist_ok=True)
            if plot == "ROC":
                save_path = plot_folder
                plot_filename = f"{target_to_predict}_ROC_{row_best_experiment['sample_type']}_{int(train_percentage * 100)}pct_{test_name}.png"
                saved_plot = os.path.join(save_path, plot_filename)
                plot_roc_curve(
                    y_test=y_test,
                    y_prob=y_prob,
                    label_encoder=label_encoder,
                    sample_type=row_best_experiment["sample_type"],
                    train_percentage=train_percentage,
                    test_accuracy=row_best_experiment["test_acc"],
                    test_name=test_name,
                    target_name=target_to_predict,
                    save_path=plot_folder,
                    mlflow_is_running=False,
                )
            elif plot == "Principal_Wavenumber":
                if not do_principal:
                    logger.info(
                        "Skipping Principal Wavenumber plot due to previous error."
                    )
                else:
                    pls_loadings = loadings.transpose()
                    wavenumber_importances = np.abs(
                        lv_importance @ pls_loadings
                    )
                    wavenumber_importances /= wavenumber_importances.sum()

                    # Remover zona da água
                    valid_mask = (wavenumbers < 1850) | (wavenumbers > 2500)
                    valid_wavenumbers = wavenumbers[valid_mask]
                    valid_importances = wavenumber_importances[valid_mask]
                    plot_wavenumber_importances(
                        valid_wavenumbers=valid_wavenumbers,
                        valid_importances=valid_importances,
                        target_name=target_to_predict,
                        sample_type=row_best_experiment["sample_type"],
                        train_percentage=train_percentage,
                        test_name=test_name,
                        group_suffix="",
                        save_path=plot_folder,
                        plot_filepath=None,
                        mlflow_is_running=False,
                    )

            elif plot == "Confusion_Matrix":
                plot_confusion_matrix(
                    y_test=y_test,
                    y_pred=y_pred,
                    label_encoder=label_encoder,
                    accuracy_score=row_best_experiment["test_acc"],
                    sample_type=sample_type,
                    train_percentage=train_percentage,
                    test_name=test_name,
                    target_name=target_to_predict,
                    threshold=None,
                    group_fam_to_use="",
                    plot_filepath=None,
                    mlflow_is_running=False,
                    save_path=plot_folder,
                )
            elif plot == "Metrics":
                saved_plot = plot_metrics_per_group(
                    experiment_results,
                    metric,
                    "sample_type",
                    target_name=target_to_predict,
                    save_path=plot_folder,
                    mlflow_is_running=False,
                    save_to_file=True,
                    bigger_is_better=best_is_max,
                )

            else:
                raise ValueError(f"Unknown plot type: {plot}")

            # Get list of all files in the project directory with the namse basename as the saved_plot
            # and replace them with the saved_plot
            for root, dirs, files in os.walk(PROJ_ROOT):
                for file in files:
                    if os.path.basename(file) == os.path.basename(saved_plot):
                        if os.path.join(root, file) == saved_plot:
                            continue
                        if os.path.exists(os.path.join(root, file)):
                            os.remove(os.path.join(root, file))
                            shutil.copyfile(
                                saved_plot, os.path.join(root, file)
                            )
