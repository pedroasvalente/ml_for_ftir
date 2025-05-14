import typer

from ml4fir.config import logger
from ml4fir.modeling.predict import predict as predict_main
from ml4fir.modeling.train import train as train_main

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
