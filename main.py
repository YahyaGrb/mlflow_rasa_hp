import os
import sys
import mlflow
import logging

from mlflow.tracking import MlflowClient
from mlflow.utils import mlflow_tags
from mlflow.entities import RunStatus
from mlflow.utils.logging_utils import eprint
from mlflow.tracking.fluent import _get_experiment_id
from rasa.shared.importers.importer import TrainingDataImporter

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
    # existing_run = _already_ran(entrypoint, parameters, git_commit)
    # if use_cache and existing_run:
    #     print(
    #         "Found existing run for entrypoint={} and parameters={}".format(entrypoint, parameters)
    #     )
    #     return existing_run
    logger.info("Launching new run for entrypoint={} and parameters={}".format(entrypoint, parameters))
    submitted_run = mlflow.run(".", entrypoint, parameters=parameters, env_manager="local", synchronous = synchronous)
    return MlflowClient().get_run(submitted_run.run_id)



def  workflow(config,training, validation):
    with mlflow.start_run() as active_run:
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
        # ratings_parquet_uri = os.path.join(etl_data_run.info.artifact_uri, "ratings-parquet-dir")

        # # We specify a spark-defaults.conf to override the default driver memory. ALS requires
        # # significant memory. The driver memory property cannot be set by the application itself.
        # als_run = _get_or_run(
        #     "als", {"ratings_data": ratings_parquet_uri, "max_iter": str(als_max_iter)}, git_commit
        # )
        # als_model_uri = os.path.join(als_run.info.artifact_uri, "als-model")

        # keras_params = {
        #     "ratings_data": ratings_parquet_uri,
        #     "als_model_uri": als_model_uri,
        #     "hidden_units": keras_hidden_units,
        # }
        # _get_or_run("train_keras", keras_params, git_commit, use_cache=False)


if __name__ == "__main__":
    training_files = "training_data.yml"
    config_file_path = "/Users/yahyaghrab/dial-once/VF scripts/VattenFall/config.yml"
    validation_files = "test_data.yml"
    import os
    os.system("spacy download fr_core_news_md") # try to get it improved and done withing venv
    workflow(config_file_path,training_files,validation_files)