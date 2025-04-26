import typer

from ml4fir.config import logger
from ml4fir.modeling.train import train as train_main

app = typer.Typer()


@app.command()
def train(
    experiment_config: str = typer.Argument(
        ..., help="Path to the experiment configuration file."
    )
):
    """
    Run the training script.
    """
    logger.info(f"Running training with config: {experiment_config}")
    train_main(experiment_config=experiment_config)


@app.command()
def predict():
    """
    Prediction using trained model
    """
    logger.info("Prediction using trained model")
