import os

import numpy as np
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

from ml4fir.modeling.config_features_importance import func_back_projection
from ml4fir.modeling.models import get_model_config
from ml4fir.modeling.results_functions import func_cv_results, results_func
from ml4fir.modeling.train_config import model_args_conf, search_args
from ml4fir.ploting import plot_confusion_matrix, plot_roc_curve

roc_plot_path = "000_ROC_plots/"
confusion_matrix_plot_path = "000_CM_plots/"
os.makedirs(roc_plot_path, exist_ok=True)
os.makedirs(confusion_matrix_plot_path, exist_ok=True)


def calculate_metrics(y_test, y_pred):
    """
    Calculate evaluation metrics.

    Parameters:
        y_test (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.

    Returns:
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

    Parameters:
        model (object): Trained model.
        x_train (array-like): Training data.
        model_type (str): Type of the model ('xgboost' or other).

    Returns:
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

    Parameters:
        y_test (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        y_prob (array-like): Predicted probabilities.
        label_encoder (object): Label encoder for decoding labels.
        sample_type (str): Type of the sample.
        train_percentage (float): Training percentage.
        model_name (str): Name of the model.
        target_column (str): Target column name.
        group_fam_to_use (optional): Optional grouping family.

    Returns:
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


def supervised_training_search(
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
    search_type="grid",  # "grid" for GridSearchCV, "bayes" for BayesSearchCV
    group_fam_to_use=None,
):
    """
    Perform supervised training using either GridSearchCV or BayesSearchCV.

    Parameters:
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
        model_type (str): Type of the model.
        search_type (str): Type of search to perform ("grid" or "bayes").
        group_fam_to_use (optional): Optional grouping family.

    Returns:
        tuple: Updated results, cross-validation results, and back projection.
    """
    config = get_model_config(model_type)
    model_name = config.desc_name

    if search_type == "grid":
        search_function_name = "GridSearchCV"
        search_fn = GridSearchCV
        param_grid = config.get_param_grid()
    elif search_type == "bayes":
        search_function_name = "BayesSearchCV"
        search_fn = BayesSearchCV
        param_grid = config.get_bayes_search_params()
        param_grid.update({"random_state": config.random_seed})
    else:
        raise ValueError("Invalid search_type. Choose 'grid' or 'bayes'.")

    # Get model and search parameters
    model_args = model_args_conf.get(model_type, {}).get(search_function_name, {})
    model = config.get_model(**model_args)

    search_params = search_args[config.name][search_function_name]
    search = search_fn(model, param_grid, **search_params)
    # Fit the search
    search.fit(x_train, y_train)
    best_model = search.best_estimator_

    # Make predictions
    y_pred = best_model.predict(x_test)
    y_prob = best_model.predict_proba(x_test)

    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred)

    # Calculate feature importances
    lv_importance = calculate_feature_importances(best_model, x_train, model_type)

    # Perform back projection
    top_wavenumbers, top_importances = func_back_projection(
        lv_importance,
        loadings,
        wavenumbers,
        metrics["test_acc"],
        target_column,
        sample_type,
        train_percentage,
        f"{model_name} ({'GridSearchCV' if search_type == 'grid' else 'BayesOpt'})",
        group_fam_to_use=group_fam_to_use,
    )

    # Update cross-validation results
    cross_validation_results = func_cv_results(
        cross_validation_results,
        sample_type,
        train_percentage,
        metrics["test_acc"],
        metrics["f1"],
        metrics["recall"],
        metrics["precision"],
        metrics["cm"],
        search,
        metrics["acc"],
        f"{model_name} ({'GridSearchCV' if search_type == 'grid' else 'BayesOpt'})",
    )

    # Generate plots
    generate_plots(
        y_test,
        y_pred,
        y_prob,
        label_encoder,
        sample_type,
        train_percentage,
        f"{model_name} ({'GridSearchCV' if search_type == 'grid' else 'BayesOpt'})",
        target_column,
        group_fam_to_use,
    )

    # Update results
    results, back_projection = results_func(
        results,
        sample_type,
        train_percentage,
        f"{model_name} ({'GridSearchCV' if search_type == 'grid' else 'BayesOpt'})",
        metrics["test_acc"],
        metrics["f1"],
        metrics["roc_auc"],
        top_wavenumbers,
        top_importances,
        back_projection,
    )

    return results, cross_validation_results, back_projection


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

    Parameters:
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

    Returns:
        tuple: Updated results, cross-validation results, and back projection.
    """
    search_to_use = search_to_use or ["grid", "bayes"]
    if not isinstance(search_to_use, list):
        search_to_use = [search_to_use]

    for search_type in search_to_use:
        results, cross_validation_results, back_projection = supervised_training_search(
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
            group_fam_to_use=group_fam_to_use,
            search_type=search_type,
        )

    # WHY: why return the results etc if they are overwritten?

    return results, cross_validation_results, back_projection
