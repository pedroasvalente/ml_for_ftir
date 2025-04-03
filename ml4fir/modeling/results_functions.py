# TODO: this whole module should be refactored, maybe it is not needed at all


def func_cv_results(
    cross_validation_results,
    sample_type,
    train_percentage,
    test_accuracy_grid,
    f1_grid,
    recall_grid,
    precision_grid,
    conf_matrix_grid,
    grid_search,
    accuracy_score,
    signature,
):
    """
    Updates the cross-validation results dictionary with the provided metrics and grid search results.

    Args:
        cross_validation_results (dict): Dictionary to store cross-validation results.
        sample_type (str): Type of sample being analyzed.
        train_percentage (float): Percentage of data used for training.
        test_accuracy_grid (float): Test accuracy from the grid search.
        f1_grid (float): F1 score from the grid search.
        recall_grid (float): Recall score from the grid search.
        precision_grid (float): Precision score from the grid search.
        conf_matrix_grid (numpy.ndarray): Confusion matrix from the grid search.
        grid_search (GridSearchCV): Fitted GridSearchCV object.
        accuracy_score (float): Overall accuracy score.
        signature (str): Model signature or identifier.

    Returns:
        dict: Updated cross-validation results dictionary.
    """
    cross_validation_results["Sample Type"].append(sample_type)
    cross_validation_results["Train Percentage"].append(train_percentage)
    cross_validation_results["Model"].append(signature)
    cross_validation_results["Balanced Accuracy"].append(float(test_accuracy_grid))
    cross_validation_results["F1 Score"].append(float(f1_grid))
    cross_validation_results["Recall"].append(float(recall_grid))
    cross_validation_results["Precision"].append(float(precision_grid))
    cross_validation_results["Confusion Matrix"].append(conf_matrix_grid.tolist())
    cross_validation_results["Best Params"].append(grid_search.best_params_)
    cross_validation_results["mean_test_score"].append(
        [float(score) for score in grid_search.cv_results_["mean_test_score"]]
    )
    cross_validation_results["std_test_score"].append(
        [float(score) for score in grid_search.cv_results_["std_test_score"]]
    )
    cross_validation_results["rank_test_score"].append(
        [int(rank) for rank in grid_search.cv_results_["rank_test_score"]]
    )
    cross_validation_results["params"].append(grid_search.cv_results_["params"])
    cross_validation_results["best_index"].append(int(grid_search.best_index_))
    for i in range(5):
        cross_validation_results[f"split{i}_test_score"].append(
            [float(score) for score in grid_search.cv_results_[f"split{i}_test_score"]]
        )
    cross_validation_results["accuracy_score"].append(accuracy_score)
    return cross_validation_results


def results_func(
    results,
    sample_type,
    train_percentage,
    model_name,
    test_accuracy,
    f1_score_val,
    roc_auc,
    top_wavenumbers,
    top_importances,
    back_projection,
):
    """
    Updates the results and back-projection dictionaries with the provided metrics and feature importances.

    Args:
        results (dict): Dictionary to store overall results.
        sample_type (str): Type of sample being analyzed.
        train_percentage (float): Percentage of data used for training.
        model_name (str): Name of the model used.
        test_accuracy (float): Test accuracy of the model.
        f1_score_val (float): F1 score of the model.
        roc_auc (float): ROC AUC score of the model.
        top_wavenumbers (list): List of top wavenumbers contributing to the model.
        top_importances (list): List of feature importances corresponding to the top wavenumbers.
        back_projection (dict): Dictionary to store back-projection data.

    Returns:
        tuple: Updated results and back-projection dictionaries.
    """
    results["Sample Type"].append(sample_type)
    results["Train Percentage"].append(train_percentage)
    results["Model"].append(model_name)
    results["Accuracy"].append(float(test_accuracy))
    results["F1 Score"].append(float(f1_score_val))
    results["ROC AUC"].append(float(roc_auc))

    n_wavenumbers = len(top_wavenumbers)
    back_projection["Sample Type"].extend([sample_type] * n_wavenumbers)
    back_projection["Train Percentage"].extend([train_percentage] * n_wavenumbers)
    back_projection["Model"].extend([model_name] * n_wavenumbers)
    back_projection["Accuracy"].extend([float(test_accuracy)] * n_wavenumbers)
    back_projection["Wavenumber (cm⁻¹)"].extend(
        [float(wn) if isinstance(wn, str) else wn for wn in top_wavenumbers]
    )
    back_projection["Importance"].extend(top_importances)
    return results, back_projection
