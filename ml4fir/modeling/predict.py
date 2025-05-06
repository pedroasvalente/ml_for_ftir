import os
import shutil

import mlflow
import pandas as pd

from ml4fir.config import (
    EXPERIMENTS_DIR,
    MLFLOW_ARTIFACTS_DIR,
    MODELS_DIR,
    main_metric,
)
from ml4fir.data import DataHandler

mlflow.autolog(disable=True)


def save_best_model(
    main_experiment,
    best_experiment,
    target_to_predict: str,
    metric: str = main_metric,
    sample_type: str = None,
):

    mlflow_best_model_path = os.path.join(
        MLFLOW_ARTIFACTS_DIR,
        str(main_experiment),
        best_experiment,
        "artifacts",
        "best model",
    )

    best_model_path = os.path.join(MODELS_DIR, target_to_predict, "best_model")
    if sample_type is not None:
        best_model_path = os.path.join(
            MODELS_DIR, target_to_predict, sample_type, "best_model"
        )
    os.makedirs(best_model_path, exist_ok=True)

    # Copy folder mlflow_best_model_path to best_model_path
    shutil.copytree(
        str(mlflow_best_model_path), str(best_model_path), dirs_exist_ok=True
    )


def check_predict_data(file_for_prediction, target_to_predict: str):
    # TODO: only needs saliva or uirne or etcs
    # Check if the example file exists
    example_path = os.path.join(EXPERIMENTS_DIR, target_to_predict, "example.csv")
    if not os.path.exists(example_path):
        raise FileNotFoundError(f"Example file not found: {example_path}")

    # Load the example data
    example_data = pd.read_csv(example_path)
    # Load the prediction data
    prediction_data = pd.read_csv(file_for_prediction)

    # Check if the columns in the prediction data match the example data
    if set(prediction_data.columns) != set(example_data.columns):
        raise ValueError("Prediction data columns do not match example data columns.")

    return True


def predict(
    file_for_prediction,
    target_to_predict: str,
    metric: str = main_metric,
    best_is_max: bool = True,
    sample_type: str = None,
    file_to_save: str = None,
):

    experiment_results_file = os.path.join(
        EXPERIMENTS_DIR, target_to_predict, "experiment_configs.csv"
    )
    experiment_results = pd.read_csv(experiment_results_file, index_col=0)
    if sample_type is not None:
        experiment_results = experiment_results[
            experiment_results["sample_type"] == sample_type
        ]

    # TODO: yeah metric as a class
    if best_is_max:
        best_experiment = experiment_results[metric].idxmax()
    else:
        best_experiment = experiment_results[metric].idxmin()

    check_predict_data(file_for_prediction, target_to_predict)

    columns_for_data = ["scale", "apply_pls", "apply_smote_resampling", "n_components"]
    row_best_experiment = experiment_results.loc[best_experiment]
    data_args = row_best_experiment[columns_for_data].to_dict()

    datahandler = DataHandler(
        data_path=file_for_prediction,
        target=target_to_predict,
        train=False,
        **data_args,
    )
    datahandler.process_sample_data(
        target=target_to_predict,
        sample_type=row_best_experiment["sample_type"],
    )

    datahandler.preprocess_data(
        train_percentage=1,
    )
    x_for_prediction = datahandler.x_train

    best_model_path = os.path.join(MODELS_DIR, target_to_predict, "best_model")
    if sample_type is not None:
        best_model_path = os.path.join(
            MODELS_DIR, target_to_predict, sample_type, "best_model"
        )
    if not os.path.exists(best_model_path):
        save_best_model(
            row_best_experiment["experiment_id"],
            best_experiment,
            target_to_predict,
            metric,
            sample_type=sample_type,
        )

    trained_model = mlflow.pyfunc.load_model(best_model_path)

    predictions = trained_model.predict(x_for_prediction)
    # BUG: pyfunc does not load the sklearn model properly
    # if hasattr(trained_model, "predict_proba"):
    #     predictions = trained_model.predict(x_for_prediction)
    #     predictions_prob = trained_model.predict_proba(x_for_prediction)
    # else:
    #     predictions_prob = trained_model.predict(x_for_prediction)
    #     predictions = np.argmax(predictions_prob, axis=-1)

    file_to_save = file_to_save or os.path.join(
        os.path.dirname(file_for_prediction), "predictions.csv"
    )
    pd.DataFrame({"predictions": predictions}).to_csv(file_to_save, index=False)

    return predictions


# predict("example.csv", target_to_predict="group_fam")
# def prediction_pipeline(file_for_prediction, target_to_predict: str, metric: str = main_metric):
#     logger.info(f"Predicting {target_to_predict} using the best model.")


#     experiment_results_file = os.path.join(
#         EXPERIMENTS_DIR, target_to_predict,"experiment_configs.csv"
#     )
#     experiment_results = pd.read_csv(experiment_results_file)


#     datahandler = DataHandler(data_path=file_for_prediction)


#     predictions = predict(x_for_prediction, target_to_predict, metric)
#     return predictions
