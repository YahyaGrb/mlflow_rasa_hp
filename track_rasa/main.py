import os
import sys
import mlflow
import logging
import click
import tempfile
sys.path.insert(1, "../src")
from utils import _transform_uri_to_path, _get_run_config
from mlflow.tracking import MlflowClient


logging.basicConfig(stream=sys.stderr, level=logging.INFO)

logger = logging.getLogger(__name__)


def _get_or_run(entrypoint, parameters, synchronous=True): #removed git commit and use_cache=True
    """
    Run the entrypoint with the given parameters.

    :param entrypoint: The entrypoint to run.
    :param parameters: The parameters to pass to the entrypoint.
    :param synchronous: Whether to run the entrypoint in a synchronous or asynchronous fashion.
    """
    logger.info("Launching new run for entrypoint={} and parameters={}".format(entrypoint, parameters))
    submitted_run = mlflow.run(".", entrypoint, parameters=parameters, synchronous = synchronous)
    succeeded = submitted_run.wait()
    return MlflowClient().get_run(submitted_run.run_id)

@click.command(help="Perform unique run of the code with provided params.")
@click.option('--params', default = {"epochs": 10, "threshold": 0.3, "model": ""})
@click.argument("config_template", default="../files/hyperopt/template_config.yml")
@click.argument("train_data", default="../files/hyperopt/training_data.yml")
@click.argument("validation_data", default="../files/hyperopt/test_data.yml")
def  workflow(params, config_template,train_data, validation_data):
    with mlflow.start_run():
        mlflow.set_experiment("track_rasa")
        with tempfile.TemporaryDirectory() as temp_config_dir:
            generated_config =_get_run_config(config_template, params, temp_config_dir)

            logger.info("Starting to train")
            train_model = _get_or_run("train", {"config":generated_config, "training":train_data})
            logger.info("Training complete")
            model_uri = os.path.join(train_model.info.artifact_uri, "model")
            logger.info(model_uri)
            model_path = _transform_uri_to_path(model_uri)
            logger.info("Starting to test")
            test_model = _get_or_run("test", {"model_path": model_path, "validation": validation_data})
            logger.info("Testing complete")


if __name__ == "__main__":
    workflow()
