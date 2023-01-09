from hyperopt import fmin, tpe, rand
from space import search_space
import os
import time
import logging
import sys
import mlflow
import numpy as np
import click
import tempfile
from urllib.parse import urlparse

from concurrent.futures import ThreadPoolExecutor
from mlflow.tracking import MlflowClient

_inf = np.finfo(np.float64).max

logging.basicConfig(stream=sys.stderr, level=logging.INFO)

logger = logging.getLogger(__name__)

def _get_or_run(entrypoint, parameters, run_id, experiment_id, synchronous=True): #removed git commit and use_cache=True
    logger.info(
        f"Launching new run for entrypoint={entrypoint} and parameters={parameters}"
    )
    submitted_run = mlflow.projects.run(".", entrypoint, parameters=parameters, env_manager="conda", synchronous = synchronous, run_id=run_id, experiment_id=experiment_id)
    succeeded = submitted_run.wait()
    return MlflowClient().get_run(submitted_run.run_id)

def _transform_uri_to_path(uri, sub_directory = ""):
    parsed_url = urlparse(uri)
    # Get the path from the parsed URL
    path = parsed_url.path
    # Make the path absolute using os.path.abspath()
    abs_path = os.path.abspath(path)
    # Get the list of files in the directory
    if sub_directory:
        files_path = os.path.join(abs_path, sub_directory)
        files = os.listdir(files_path)
        # Get the first file in the list (assuming there is at least one file in the directory)
        file = files[0]
        # Create the new path by combining the file name with the current working directory
        return os.path.join(files_path, file)
    return abs_path


@click.command(help="Perform hyperparameter search with Hyperopt library. Optimize dl_train target.")
@click.argument("config_template", default="../files/hyperopt/template_config.yml")
@click.argument("train_data", default="../files/hyperopt/training_data.yml")
@click.argument("validation_data", default="../files/hyperopt/test_data.yml")
@click.option("--max-runs", type=click.INT, default=10, help="Maximum number of runs to evaluate.")
@click.option("--metric", type=click.STRING, default="f1_intent", help="Metric to optimize on.")
@click.option("--algo", type=click.STRING, default="tpe.suggest", help="Optimizer algorithm.")
def workflow(config_template, train_data, validation_data,max_runs, metric, algo):
    """
    Hyperparameter search with Hyperopt library.
    Args:
        config_template: Path to the template configuration file.
        train_data: Path to the training data file.
        validation_data: Path to the validation data file.
        max_runs: Maximum number of runs to evaluate.
        metric: Metric to optimize on.
        algo: Optimizer algorithm.
    Return:
        None
    """

    start_time = time.time()

    def new_eval(experiment_id):
        """
        Create a new eval function
        :experiment_id: Experiment id for the training run
        :return: new eval function.
        """
        def eval(space):
            """
            Train then evaluate ml flows run with hyperopt optimization according to a search space.
            Args:
                space (Dict): hyper opt search space
            Return:
                test_loss (Float): test loss for the metric specified
            """

            with mlflow.start_run(nested=True) as child_run:
                logger.info(f"Search space: {space}")

                # Creating temporary config file containing space variables
                with tempfile.TemporaryDirectory() as temp_config_dir:
                    generated_config = f"{temp_config_dir}/run_config.yml"
                    config_template_path = os.path.relpath(config_template)
                    train_data_path = os.path.relpath(train_data)
                    validation_data_path = os.path.relpath(validation_data)

                    with open(config_template_path) as f:
                        template_config_yml = f.read().format(**space)
                        with open(generated_config, 'w+') as temp_f:
                            temp_f.write(template_config_yml)

                    logger.info("Starting to train")
                    train_model = _get_or_run("train", {"config":generated_config, "training":train_data_path}, child_run.info.run_id,experiment_id)
                    logger.info("Training complete")
                    model_path = _transform_uri_to_path(train_model.info.artifact_uri, "model")
                    logger.info("Starting to test")
                    test_model = _get_or_run("test", {"model_path": model_path, "validation": validation_data_path}, child_run.info.run_id,experiment_id)
                    logger.info("Testing complete with the following validation metrics:")
                    metrics = test_model.data.metrics
                    logger.info(metrics)
                    test_loss = 1 - metrics[metric]
                    mlflow.log_params(params = space)
            return test_loss
        return eval

    with mlflow.start_run() as run:
        experiment_id = run.info.experiment_id
        # activate multithreading
        runs = [(np.random.uniform(1e-5, 1e-1), np.random.uniform(0, 1.0)) for _ in range(max_runs)]
        with ThreadPoolExecutor(max_workers=max_runs) as executor: # use max_runs for number of workers/threads
            executor.map(
                fmin(
                    fn=new_eval(experiment_id),
                    space=search_space,
                    algo=tpe.suggest if algo == "tpe.suggest" else rand.suggest,
                    max_evals=max_runs),
                runs,
            )
            logger.info("Hyperparameter search complete!")
            # find the best run, log its metrics as the final metrics of this run.
            client = MlflowClient()
            runs = client.search_runs(
                [experiment_id], f"tags.mlflow.parentRunId = '{run.info.run_id}' ")
            best_test_loss = _inf
            best_run = None
            for r in runs:
                if 1 - r.data.metrics[metric] < best_test_loss:
                    best_run = r
                    best_test_loss = 1 - r.data.metrics[metric]
            best_config_path = _transform_uri_to_path(best_run.info.artifact_uri, "config")
            with open(best_config_path) as f:
                config_yml = f.read()
                logger.info("The best configuration is: \n{}\n".format(config_yml))
            mlflow.set_tag("best_run", best_run.info.run_id)
            mlflow.log_metrics(
                {
                    "Best_{}_loss".format(metric): best_test_loss,
                }
            )
            mlflow.log_artifacts(local_dir=_transform_uri_to_path(best_run.info.artifact_uri))
    logger.info(f"Total execution time: {time.time() - start_time} seconds")



if __name__ == "__main__":
    workflow()