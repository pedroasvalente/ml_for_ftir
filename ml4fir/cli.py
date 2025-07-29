import typer
import os
import pandas as pd
import shutil
from pathlib import Path

from ml4fir.config import logger
from ml4fir.modeling.predict import predict as predict_main
from ml4fir.modeling.train import train as train_main
from ml4fir.ploting.plots import plot as plot_main

app = typer.Typer()


@app.command()
def train(
    experiment_config: str = typer.Argument(
        ..., help="Path to the experiment configuration file."
    ),
):
    """
    Run the training script.
    """
    logger.info(f"Running training with config: {experiment_config}")
    train_main(experiment_config=experiment_config)



@app.command()
def clear_runs():
    """
    For each experiment_configs.csv under experiments/*/,
    keeps only mlartifacts/{run_id} and mlruns/{run_id} folders that are present in the CSV run_id column.
    Removes all others for each experiment_id.
    """
    base_dir = Path("experiments")
    for csv_path in base_dir.glob("*/experiment_configs.csv"):
        df = pd.read_csv(csv_path)
        run_ids = set(df["run_id"].dropna().astype(str))
        experiment_ids = set(df["experiment_id"].dropna().astype(str))

        for experiment_id in experiment_ids:
            # the experiment csv was not complete
            if experiment_id == "450165946480039955":
                continue
            for folder_type in ["mlartifacts", "mlruns"]:
                exp_folder = Path(folder_type) / experiment_id
                if exp_folder.exists():
                    for run_folder in exp_folder.iterdir():
                        if run_folder.is_dir() and run_folder.name not in run_ids:
                            print(f"Deleting {run_folder}")
                            shutil.rmtree(run_folder)


@app.command()
def predict(
    file_for_prediction: str = typer.Argument(
        ..., help="Path to the file containing data for prediction."
    ),
    target_to_predict: str = typer.Option(
        ..., help="Target variable to predict."
    ),
    sample_type: str = typer.Option(
        None, help="Sample type (e.g., saliva, urine, etc.)."
    ),
):
    """
    Prediction using trained model.
    """
    logger.info(
        f"Prediction using trained model for target: {target_to_predict}"
    )
    logger.info(f"File for prediction: {file_for_prediction}")
    if sample_type:
        logger.info(f"Sample type: {sample_type}")
    predict_main(
        file_for_prediction=file_for_prediction,
        target_to_predict=target_to_predict,
        sample_type=sample_type,
    )


@app.command()
def plot(
    target_to_predict: str = typer.Option(
        ..., help="Target variable to predict."
    ),
    sample_type: str = typer.Option(
        None, help="Sample type (e.g., saliva, urine, etc.)."
    ),
    plots_to_do: str = typer.Option(
        None, help="Comma-separated list of plots to generate."
    ),
    only_best: bool = typer.Option(
        True, help="Whether to plot only the best experiment."
    ),
):
    """
    Prediction using trained model.
    """
    plots_to_do = (
        plots_to_do or "ROC,Confusion_Matrix,Principal_Wavenumber,Metrics"
    )
    plots_to_do = plots_to_do.split(",")
    logger.info(f"Doing plots: {plots_to_do}")
    logger.info(f"For target: {target_to_predict}")
    if sample_type:
        logger.info(f"Sample type: {sample_type}")

    plot_main(
        target_to_predict=target_to_predict,
        plots_to_do=plots_to_do,
        only_best=only_best,
        sample_type=sample_type,
    )
