import tempfile
import mlflow
import click
import time


from rasa.shared.data import TrainingType
from rasa.model_training import train_nlu


def personnalised_trainer(config, training, training_type=TrainingType.NLU): #for future use only
    from rasa.shared.importers.importer import TrainingDataImporter
    from rasa.engine.recipes.recipe import Recipe
    from rasa.model_training import (
        DaskGraphRunner,
        GraphTrainer, # really faster using cache
        LocalTrainingCache,
        Path,
        _create_model_storage,
        _determine_model_name,
    )
    start_time = time.time()
    file_importer = TrainingDataImporter.load_from_config(
            config_path = config, training_data_paths = training
    )
    configuration = file_importer.get_config()
    recipe = Recipe.recipe_for_name(configuration.get("recipe"))
    model_configuration = recipe.graph_config_for_recipe(
        configuration,
        cli_parameters={},
        training_type=training_type,
    )

    with tempfile.TemporaryDirectory() as temp_model_dir:
        model_storage = _create_model_storage(
            is_finetuning=False,
            model_to_finetune=None,
            temp_model_dir=Path(temp_model_dir),
        )
        cache = LocalTrainingCache()
        trainer = GraphTrainer(model_storage, cache, DaskGraphRunner)

        model_name = _determine_model_name(
            fixed_model_name=None, training_type=training_type
        )
        full_model_path = Path(temp_model_dir, model_name)
        trainer.train(
            model_configuration,
            file_importer,
            full_model_path,
            force_retraining=False,
            is_finetuning=False,
        )
        duration = time.time() - start_time
        print(f"Training time: {duration}")
        mlflow.log_artifact(model_path, artifact_path="model")
        return duration

def fast_trainer(config, training):
    mlflow.set_experiment("Trains")
    with tempfile.TemporaryDirectory() as temp_model_dir:
        start_time = time.time()
        model_path = train_nlu(config=config, nlu_data=training, output=temp_model_dir, additional_arguments = {}) #faster
        duration =  time.time() - start_time
        print(f"Training time: {duration}")
        mlflow.log_artifact(model_path, artifact_path="model")
        return duration



# @click.command()
# @click.option("--config", help="Path readable by Spark to the ratings Parquet file")
# @click.option("--training", help="Path readable by Spark to the ratings Parquet file")
def train(config, training):
    duration = fast_trainer(config, training)
    mlflow.log_metric("train_duration", duration)
    mlflow.log_artifact(config, artifact_path="config")

if __name__ == '__main__':
    # train()
    # for manual execution uncomment following and comment preceding
    config_file_path = "config.yml"
    training_files = "training_data.yml"

    train(config_file_path,training_files) # replace with train_nlu and see which is faster
    print("train completed")
