from hyperopt import fmin, tpe, rand, space_eval
from space import search_space
import os
import logging
import sys
import mlflow
import numpy as np
import click
from urllib.parse import urlparse


import mlflow.projects
from mlflow.tracking import MlflowClient

_inf = np.finfo(np.float64).max

logging.basicConfig(stream=sys.stderr, level=logging.INFO)

logger = logging.getLogger(__name__)

def _get_or_run(entrypoint, parameters, run_id, experiment_id, synchronous=True): #removed git commit and use_cache=True
    logger.info("Launching new run for entrypoint={} and parameters={}".format(entrypoint, parameters))
    submitted_run = mlflow.run(".", entrypoint, parameters=parameters, env_manager="local", synchronous = synchronous, run_id=run_id, experiment_id=experiment_id)
    succeeded = submitted_run.wait()
    return MlflowClient().get_run(submitted_run.run_id)

def _transform_uri_to_path(uri):
    parsed_url = urlparse(uri)
    # Get the path from the parsed URL
    path = parsed_url.path
    # Make the path absolute using os.path.abspath()
    abs_path = os.path.abspath(path)
    # Get the list of files in the directory
    files = os.listdir(abs_path)
    # Get the first file in the list (assuming there is at least one file in the directory)
    file = files[0]
    # Create the new path by combining the file name with the current working directory
    return os.path.join(abs_path, file)


@click.command(help="Perform hyperparameter search with Hyperopt library. Optimize dl_train target.")
@click.argument("config_template", default="../files/template_config.yml")
@click.argument("train_data", default="../files/training_data.yml")
@click.argument("validation_data", default="../files/test_data.yml")
@click.option("--max-runs", type=click.INT, default=10, help="Maximum number of runs to evaluate.")
@click.option("--metric", type=click.STRING, default="f1-intent", help="Metric to optimize on.")
@click.option("--algo", type=click.STRING, default="tpe.suggest", help="Optimizer algorithm.")

def workflow(config_template, train_data, validation_data,max_runs, metric, algo):
    """
    Run hyperparameter optimization.
    """
    # create random file to store run ids of the training tasks
    # tracking_client = MlflowClient()

    def new_eval(experiment_id):
        """
        Create a new eval function

        :param nepochs: Number of epochs to train the model.
        :experiment_id: Experiment id for the training run
        :valid_null_loss: Loss of a null model on the validation dataset
        :test_null_loss: Loss of a null model on the test dataset.
        :return_test_loss: Return both validation and test loss if set.

        :return: new eval function.
        """
        def eval(space):
            """
            Train Keras model with given parameters by invoking MLflow run.

            Notice we store runUuid and resulting metric in a file. We will later use these to pick
            the best run and to log the runUuids of the child runs as an artifact. This is a
            temporary workaround until MLflow offers better mechanism of linking runs together.

            :param params: Parameters to the train_keras script we optimize over:
                          learning_rate, drop_out_1
            :return: The metric value evaluated on the validation data.
            """
            import mlflow.tracking
            with mlflow.start_run(nested=True) as child_run:
                logger.info(f"Search space: {space}")

                # Creating temporary config file containing space variables
                os.makedirs('./tmp_configs', exist_ok=True)
                config_path = "./tmp_configs/run_config.yml"
                config_template_path = os.path.relpath(config_template)
                train_data_path = os.path.relpath(train_data)
                validation_data_path = os.path.relpath(validation_data)

                with open(config_template_path) as f:
                    template_config_yml = f.read().format(**space)
                    with open(config_path, 'w+') as temp_f:
                        temp_f.write(template_config_yml)

                logger.info("Starting to train")
                train_model = _get_or_run("train", {"config":config_path, "training":train_data_path}, child_run.info.run_id,experiment_id)
                logger.info("Training complete")
                model_uri = os.path.join(train_model.info.artifact_uri, "model")
                logger.info(model_uri)
                model_path = _transform_uri_to_path(model_uri)
                logger.info("Starting to test")
                test_model = _get_or_run("test", {"model_path": model_path, "validation": validation_data_path}, child_run.info.run_id,experiment_id)
                logger.info("Testing complete")
                metrics = test_model.data.metrics
                print(metrics)
                test_loss = 1 - metrics[metric]
                mlflow.log_params(params = space)
            return test_loss

            # if succeeded:
            #     training_run = tracking_client.get_run(p.run_id)
            #     # metrics = training_run.data.metrics
            # else:
            #     # run failed => return null loss
            #     tracking_client.set_terminated(p.run_id, "FAILED")
        return eval

    with mlflow.start_run() as run:
        experiment_id = run.info.experiment_id
        # Evaluate null model first.
        best = fmin(
            fn=new_eval(experiment_id),
            space=search_space,
            algo=tpe.suggest if algo == "tpe.suggest" else rand.suggest,
            max_evals=max_runs,
        )
        mlflow.set_tag("best params", str(best))
        logger.info("Hyperparameter search complete!")

        best_config = space_eval(search_space, best)
        logger.info(f"The best values are: {best_config}")

        with open(config_template) as f:
            config_yml = f.read().format(**best_config)
            logger.info("The best configuration is: \n{}\n".format(config_yml))
        # find the best run, log its metrics as the final metrics of this run.
        client = MlflowClient()
        runs = client.search_runs(
            [experiment_id], "tags.mlflow.parentRunId = '{run_id}' ".format(run_id=run.info.run_id)
        )
        best_test_loss = _inf
        best_run = None
        for r in runs:
            if r.data.metrics[metric] < best_test_loss:
                best_run = r
                best_test_loss = r.data.metrics[metric]
        mlflow.set_tag("best_run", best_run.info.run_id)
        mlflow.log_metrics(
            {
                "test_{}".format(metric): best_test_loss,
            }
        )



if __name__ == "__main__":
    import os
    # os.system("spacy download fr_core_news_md") # try to get it improved and done withing venv
    # os.system("spacy download fr_core_news_sm") # try to get it improved and done withing venv
    # os.system("spacy download fr_dep_news_trf") # try to get it improved and done withing venv
    workflow()