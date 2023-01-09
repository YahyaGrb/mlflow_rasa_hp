from rasa.model_training import train_nlu

import tempfile
import mlflow
import mlflow.projects
import click
import time


def fast_trainer(config, training):
    with tempfile.TemporaryDirectory() as temp_model_dir:
        start_time = time.time()
        model_path = train_nlu(
            config=config,
            nlu_data=training,
            output=temp_model_dir,
            additional_arguments={},
        )  # faster
        duration = time.time() - start_time
        print(f"Training time: {duration}")
        mlflow.log_artifact(model_path, artifact_path="model")
        return duration


@click.command()
@click.option("--config", help="Path pointing to the config.yml file")
@click.option("--training", help="Path pointing to nlu training yml file")
def train(config, training):
    duration = fast_trainer(config, training)
    mlflow.log_metric("train_duration", duration)
    mlflow.log_artifact(config, artifact_path="config")


if __name__ == "__main__":
    train()
    # # for manual execution uncomment following and comment preceding
    # config_file_path = "../files/config.yml"
    # training_files = "../files/training_data.yml"

    # train(config_file_path,training_files) # replace with train_nlu and see which is faster
    # print("train completed")
