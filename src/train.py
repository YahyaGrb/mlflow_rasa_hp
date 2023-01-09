from rasa.model_training import train_nlu

import tempfile
import mlflow
import click
import time




@click.command()
@click.option("--config", help="Path pointing to the config.yml file")
@click.option("--training", help="Path pointing to nlu training yml file")
def train(config, training):
    """
    Trains a rasa NLU model using the given configuration and training file path.
    It also logs in mlflow the training duration as a metric and the model as an artifact.

    Args:
        config (string): Path to the config file (config.yml) for NLU.
        training (string): Path to the NLU training data (training_data.yml).
    Returns:
        None
    """

    with tempfile.TemporaryDirectory() as temp_model_dir:
        start_time = time.time()
        model_path = train_nlu(
            config=config,
            nlu_data=training,
            output=temp_model_dir,
            additional_arguments={},
        )
        duration = time.time() - start_time
        print(f"Train completed.\nTraining time: {duration} s")
        mlflow.log_metric("train_duration", duration)
        mlflow.log_artifact(model_path, artifact_path="model")
        mlflow.log_artifact(config, artifact_path="config")


if __name__ == "__main__":
    train()
