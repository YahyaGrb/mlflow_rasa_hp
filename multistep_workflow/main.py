import os
import sys
import mlflow
import logging

from mlflow.tracking import MlflowClient

from urllib.parse import urlparse

logging.basicConfig(stream=sys.stderr, level=logging.INFO)

logger = logging.getLogger(__name__)

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


def _get_or_run(entrypoint, parameters, synchronous=True): #removed git commit and use_cache=True
    logger.info("Launching new run for entrypoint={} and parameters={}".format(entrypoint, parameters))
    submitted_run = mlflow.run(".", entrypoint, parameters=parameters, env_manager="local", synchronous = synchronous)
    succeeded = submitted_run.wait()
    return MlflowClient().get_run(submitted_run.run_id), succeeded



def  workflow(config,training, validation):
    with mlflow.start_run() as active_run:
        mlflow.set_experiment("MLProject workflow full experiment")

        # os.environ["SPARK_CONF_DIR"] = os.path.abspath(".")
        # git_commit = active_run.data.tags.get(mlflow_tags.MLFLOW_GIT_COMMIT)
        logger.info("Starting to train")
        train_model = _get_or_run("train", {"config":config, "training":training})
        logger.info("Training complete")
        model_uri = os.path.join(train_model.info.artifact_uri, "model")
        logger.info(model_uri)
        model_path = _transform_uri_to_path(model_uri)
        logger.info("Starting to test")
        test_model = _get_or_run("test", {"model_path": model_path, "validation": validation})
        logger.info("Testing complete")


if __name__ == "__main__":
    training_files = "../files/training_data.yml"
    config_file_path = "../files/config.yml"
    validation_files = "../files/test_data.yml"
    import os
    os.system("spacy download fr_core_news_md") # try to get it improved and done withing venv
    workflow(config_file_path,training_files,validation_files)
