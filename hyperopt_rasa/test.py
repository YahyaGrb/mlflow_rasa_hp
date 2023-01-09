"""
This script will serve to run the tests on the created model

"""
import json
import time
import tempfile
import asyncio
import mlflow
import os

from rasa.model_testing import test_nlu

def extract_metric(file, metric_name):
    with open(file, "r") as f:
        metrics = json.load(f)

    return metrics["weighted avg"][metric_name]


import click
@click.command()
@click.option("--model_path", help="Path txt pointing to the last trained model in mlrun files")
@click.option("--validation", help="Path pointing to the validation files")
def evaluate_nlu_model(model_path, validation):
    """
    Evaluates a trained Rasa NLU model on a set of NLU validation data.
    Args:
    model_path: Path to the last trained model in mlrun files
    validation: Path pointing to the validation files
    """
    start_time = time.time()

    with tempfile.TemporaryDirectory() as temp_results_dir:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            test_nlu( # faster than both compute_metrics, run_evaluation
            model=model_path,
            nlu_data=validation,
            output_directory=temp_results_dir,
            additional_arguments={},
            ))
        if not os.listdir(temp_results_dir):
            print("Test failed. No results have been generated")
            metrics = {
                    "support": 0,
                    "f1_intent": 0,
                    "f1_entity": 0,
                    "elapsed_time": time.time() - start_time,
                    }
            mlflow.log_metrics(metrics = metrics)
            return metrics
        support = extract_metric(f"{temp_results_dir}/intent_report.json","support")
        f1_intent = extract_metric(f"{temp_results_dir}/intent_report.json","f1-score")
        try:
            f1_entity = extract_metric(f"{temp_results_dir}/DIETClassifier_report.json","f1-score") # if it crashes, you probably didn't use entities, either add entities or remove this line
        except FileNotFoundError:
            print("Unable to log entity f1. You need to provide entities tagging within your examples")
            f1_entity = 0
        mlflow.log_artifacts(temp_results_dir,  artifact_path="reports")

    end_time = time.time()
    print(f"Training time: {end_time - start_time}")
    metrics = {
            "support": support,
            "f1_intent": f1_intent,
            "f1_entity": f1_entity,
            "elapsed_time": end_time - start_time,
            }
    mlflow.log_metrics(metrics = metrics)
    return metrics

if __name__=="__main__":
    evaluate_nlu_model()
