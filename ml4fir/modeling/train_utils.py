import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV

from ml4fir.config import global_threshold
from ml4fir.modeling.models_experiment_conf import models_experiment
from ml4fir.modeling.results_functions import results_func
from ml4fir.modeling.train_config import model_args_conf, search_args
from ml4fir.ploting import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_wavenumber_importances,
)

roc_plot_path = "000_ROC_plots/"
confusion_matrix_plot_path = "000_CM_plots/"
os.makedirs(roc_plot_path, exist_ok=True)
os.makedirs(confusion_matrix_plot_path, exist_ok=True)


def get_principal_wavenumber_path(target_name, group_fam_to_use=None):
    folder = f"000_principal_wavenumber/{target_name}"
    if group_fam_to_use:
        folder += f"_{group_fam_to_use}"
    os.makedirs(folder, exist_ok=True)
    return folder


def calculate_metrics(y_test, y_pred):
    """
    Calculate evaluation metrics.

    Parameters
    ----------
        y_test (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.

    Returns
    -------
        dict: A dictionary containing calculated metrics.
    """
    metrics = {
        "test_acc": balanced_accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "cm": confusion_matrix(y_test, y_pred),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "acc": accuracy_score(y_test, y_pred),
        "roc_auc": None,  # Placeholder for ROC AUC, calculated later
    }
    return metrics


def calculate_feature_importances(model, x_train, model_type):
    """
    Calculate feature importances for the given model.

    Parameters
    ----------
        model (object): Trained model.
        x_train (array-like): Training data.
        model_type (str): Type of the model ('xgboost' or other).

    Returns
    -------
        np.ndarray: An array of feature importances.
    """
    if model_type == "xgboost":
        booster = model.get_booster()
        importance_dict = booster.get_score(importance_type="gain")
        lv_importance = np.zeros(x_train.shape[1])
        for i in range(len(lv_importance)):
            key = f"f{i}"
            if key in importance_dict:
                lv_importance[i] = importance_dict[key]
        lv_importance /= lv_importance.sum()
    else:
        lv_importance = model.feature_importances_

    return lv_importance


def generate_plots(
    y_test,
    y_pred,
    y_prob,
    label_encoder,
    sample_type,
    train_percentage,
    model_name,
    target_column,
    group_fam_to_use=None,
):
    """
    Generate confusion matrix and ROC curve plots.

    Parameters
    ----------
        y_test (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        y_prob (array-like): Predicted probabilities.
        label_encoder (object): Label encoder for decoding labels.
        sample_type (str): Type of the sample.
        train_percentage (float): Training percentage.
        model_name (str): Name of the model.
        target_column (str): Target column name.
        group_fam_to_use (optional): Optional grouping family.

    Returns
    -------
        None
    """
    plot_confusion_matrix(
        y_test,
        y_pred,
        label_encoder,
        balanced_accuracy_score(y_test, y_pred),
        sample_type,
        train_percentage,
        model_name,
        target_column,
        group_fam_to_use=group_fam_to_use,
    )

    plot_roc_curve(
        y_test,
        y_prob,
        label_encoder,
        sample_type,
        train_percentage,
        balanced_accuracy_score(y_test, y_pred),
        model_name,
        target_column,
        group_fam_to_use=group_fam_to_use,
    )


def perform_model_search(
    x_train,
    y_train,
    model_type,
    search_type,
):
    """
    Perform model search (GridSearchCV or BayesSearchCV), fit the model, and evaluate it.

    Parameters
    ----------
        x_train (array-like): Training features.
        y_train (array-like): Training labels.
        model_type (str): Type of the model ('random_forest', 'mlp', 'decision_tree', 'xgboost').
        search_type (str): Type of search to perform ("grid" or "bayes").

    Returns
    -------
        tuple: Best model, predictions, probabilities, metrics, and feature importances.
    """
    # Get model configuration
    # TODO: we need to take this to the upper level.
    config = models_experiment[model_type]
    model_name = config.desc_name
    param_search_space = config.get_params(search_type)

    # Determine search function and parameters
    if search_type == "GridSearchCV":
        search_fn = GridSearchCV
    elif search_type == "BayesSearchCV":
        search_fn = BayesSearchCV
    else:
        raise ValueError(
            "Invalid search_type. Choose 'GridSearchCV' or 'BayesSearchCV'."
        )

    # Get model and search parameters
    model_args = model_args_conf.get(model_type, {}).get(search_type, {})
    model = config.get_model(**model_args)
    search_params = search_args[config.name][search_type]

    # Perform search
    search = search_fn(model, param_search_space, **search_params)
    search.fit(x_train, y_train)
    # best_model = search.best_estimator_
    # TODO: this function gots to save the models, maybe just the best, and maybe using mlflow
    return search


def evaluate_model(best_model, x_test, y_test, x_train, model_type):
    """
    Evaluate the model by making predictions, calculating metrics, and feature importances.

    Parameters
    ----------
        best_model (object): Trained model.
        x_test (array-like): Testing features.
        y_test (array-like): Testing labels.
        x_train (array-like): Training features.
        model_type (str): Type of the model ('random_forest', 'mlp', 'decision_tree', 'xgboost').

    Returns
    -------
        tuple: Predictions, probabilities, metrics, and feature importances.
    """
    # Make predictions
    y_pred = best_model.predict(x_test)
    y_prob = best_model.predict_proba(x_test)

    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred)

    # Calculate feature importances
    lv_importance = calculate_feature_importances(best_model, x_train, model_type)

    return y_pred, y_prob, metrics, lv_importance


def save_wavenumber_importances(
    valid_wavenumbers,
    valid_importances,
    target_name,
    sample_type,
    train_percentage,
    test_name,
    test_accuracy,
    group_suffix,
    save_path,
):
    """
    Save wavenumber importances to an Excel file.

    Parameters
    ----------
        valid_wavenumbers (array-like): Array of valid wavenumbers.
        valid_importances (array-like): Array of corresponding importances.
        target_name (str): Name of the target variable.
        sample_type (str): Type of the sample.
        train_percentage (float): Training percentage (e.g., 0.8 for 80%).
        test_name (str): Name of the test or model configuration.
        test_accuracy (float): Test accuracy of the model.
        group_suffix (str): Suffix for filenames (e.g., based on group family).
        save_path (str): Path to save the Excel file.

    Returns
    -------
        str: Path to the saved Excel file.
    """
    # Create DataFrame
    df_out = pd.DataFrame(
        {"Wavenumber (cm⁻¹)": valid_wavenumbers, "Importance": valid_importances}
    )

    # Construct Excel filename and filepath
    # WHY: pq que raio excel e nao csv????
    excel_filename = (
        f"{target_name}_wavenumbers_importance_{sample_type}_"
        f"{int(train_percentage * 100)}pct_{test_name}_accuracy_"
        f"{test_accuracy:.4f}{group_suffix}.xlsx"
    )
    excel_filepath = os.path.join(save_path, excel_filename)

    # Save to Excel
    df_out.to_excel(excel_filepath, index=False)
    print(f"Excel file saved to: {excel_filepath}")


    csv_filename = (
        f"{target_name}_wavenumbers_importance_{sample_type}_"
        f"{int(train_percentage * 100)}pct_{test_name}_accuracy_"
        f"{test_accuracy:.4f}{group_suffix}.xlsx"
    )
    csv_filepath = os.path.join(save_path, csv_filename)

    # Save to csv
    df_out.to_csv(csv_filepath, index=False)
    # TODO: make the prints inside the logger
    print(f"CSV file saved to: {csv_filepath}")



    return excel_filepath


def supervised_training(
    x_train,
    y_train,
    x_test,
    y_test,
    label_encoder,
    sample_type,
    train_percentage,
    loadings,
    wavenumbers,
    results,
    cross_validation_results,
    target_column,
    back_projection,
    model_type,
    group_fam_to_use=None,
    search_to_use=None,  # "grid" for GridSearchCV, "bayes" for BayesSearchCV
):
    """
    Train a supervised model based on the specified model_type.

    Parameters
    ----------
        x_train (array-like): Training features.
        y_train (array-like): Training labels.
        x_test (array-like): Testing features.
        y_test (array-like): Testing labels.
        label_encoder (object): Label encoder for decoding labels.
        sample_type (str): Type of the sample.
        train_percentage (float): Training percentage.
        loadings (array-like): Loadings for back projection.
        wavenumbers (array-like): Wavenumbers for back projection.
        results (dict): Results dictionary to update.
        cross_validation_results (dict): Cross-validation results dictionary to update.
        target_column (str): Target column name.
        back_projection (dict): Back projection dictionary to update.
        model_type (str): The type of model to train. Options are:
                          'random_forest', 'mlp', 'decision_tree', 'xgboost'.
        group_fam_to_use (optional): Optional grouping family.

    Returns
    -------
        tuple: Updated results, cross-validation results, and back projection.
    """
    search_to_use = search_to_use or ["grid", "bayes"]
    if not isinstance(search_to_use, list):
        search_to_use = [search_to_use]

    for search_type in search_to_use:

        # TODO: lazy fuck, think this better
        if search_type == "grid":
            search_type = "GridSearchCV"
        elif search_type == "bayes":
            search_type = "BayesSearchCV"
        else:
            raise ValueError("Invalid search_type. Choose 'grid' or 'bayes'.")

        # Perform search for best model, by training all configuration in models_experiment_conf
        search = perform_model_search(x_train, y_train, model_type, search_type)

        best_model = search.best_estimator_

        # Evaluation
        # TODO: this most likely does not need to be a function, but we might separete into the diferent evals there.
        y_pred, y_prob, metrics, lv_importance = evaluate_model(
            best_model, x_test, y_test, x_train, model_type
        )

        # Get model configuration
        config = models_experiment[model_type]
        model_name = config.desc_name

        test_name = f"{model_name} ({search_type})"

        # TODO: write better coments, and in english.
        # Transpor os loadings
        pls_loadings = loadings.transpose()
        wavenumber_importances = np.abs(lv_importance @ pls_loadings)
        wavenumber_importances /= wavenumber_importances.sum()

        # Remover zona da água
        valid_mask = (wavenumbers < 1850) | (wavenumbers > 2500)
        valid_wavenumbers = wavenumbers[valid_mask]
        valid_importances = wavenumber_importances[valid_mask]

        top_indices = np.argsort(valid_importances)[-20:][::-1]
        top_wavenumbers = valid_wavenumbers[top_indices]
        top_importances = valid_importances[top_indices]

        test_accuracy = metrics["test_acc"]

        # TODO: check these paths.
        # Determine group suffix and save path
        group_suffix = f"_{group_fam_to_use}" if group_fam_to_use else ""
        save_path = get_principal_wavenumber_path(target_column, group_fam_to_use)

        # Guardar Excel
        if test_accuracy >= global_threshold / 100:
            save_wavenumber_importances(
                valid_wavenumbers,
                valid_importances,
                target_column,
                sample_type,
                train_percentage,
                test_name,
                test_accuracy,
                group_suffix,
                save_path,
            )

            plot_wavenumber_importances(
                valid_wavenumbers,
                valid_importances,
                target_column,
                sample_type,
                train_percentage,
                test_name,
                group_suffix,
                save_path,
            )

        f1_grid = metrics["f1"]
        recall_grid = metrics["recall"]
        precision_grid = metrics["precision"]
        conf_matrix_grid = metrics["cm"]
        accuracy = metrics["acc"]

        # separate the results for the best model and the per experiment results, just save for now.
        best_model_results = {
            "Sample Type": sample_type,
            "Train Percentage": train_percentage,
            "Model": test_name,
            "Balanced Accuracy": float(test_accuracy),
            "F1 Score": float(f1_grid),
            "Recall": float(recall_grid),
            "Precision": float(precision_grid),
            "Confusion Matrix": conf_matrix_grid.tolist(),
            # "Best Params":search.best_params_,
            **search.best_params_,
        }

        grid_search_results = pd.DataFrame(search.cv_results_)
        all_grid_params = grid_search_results["params"].to_list()
        all_grid_params = pd.DataFrame(all_grid_params)
        grid_search_results.drop(columns=["params"], inplace=True)
        grid_search_results = grid_search_results.join(all_grid_params)

        cross_validation_results = {
            "Sample Type": sample_type,
            "Train Percentage": train_percentage,
            "Model": test_name,
            "Balanced Accuracy": float(test_accuracy),
            "F1 Score": float(f1_grid),
            "Recall": float(recall_grid),
            "Precision": float(precision_grid),
            "Confusion Matrix": conf_matrix_grid.tolist(),
            "Best Params": search.best_params_,
            "mean_test_score": [
                float(f) for f in search.cv_results_["mean_test_score"]
            ],
            "std_test_score": [float(f) for f in search.cv_results_["std_test_score"]],
            "rank_test_score": [int(f) for f in search.cv_results_["rank_test_score"]],
            "params": [f for f in search.cv_results_["params"]],
            "best_index": int(search.best_index_),
            "accuracy_score": accuracy,
        }
        for i in range(5):
            cross_validation_results[f"split{i}_test_score"] = [
                float(f) for f in search.cv_results_[f"split{i}_test_score"]
            ]

        # WHY: Aqui esta a meter no mesmo saco duas coisas diferentes.
        # estas a guardar no mesmo sitio os resultados individuas (de cada parametro das searchs, aquelas lists) e o resuktado do melhor modelo daquela search.abs
        # Isto devia de ser dois diferentes. TODO
        # Update cross-validation results
        # cross_validation_results = func_cv_results(
        #     cross_validation_results,
        #     sample_type,
        #     train_percentage,
        #     metrics["test_acc"],
        #     metrics["f1"],
        #     metrics["recall"],
        #     metrics["precision"],
        #     metrics["cm"],
        #     search,
        #     metrics["acc"],
        #     test_name,
        # )

        # Generate plots
        generate_plots(
            y_test,
            y_pred,
            y_prob,
            label_encoder,
            sample_type,
            train_percentage,
            test_name,
            target_column,
            group_fam_to_use,
        )

        # Update results
        results, back_projection = results_func(
            results,
            sample_type,
            train_percentage,
            test_name,
            metrics["test_acc"],
            metrics["f1"],
            metrics["roc_auc"],
            top_wavenumbers,
            top_importances,
            back_projection,
        )

    # WHY: why return the results etc if they are overwritten?

    return results, cross_validation_results, back_projection
