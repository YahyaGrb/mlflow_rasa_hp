"""
This script will serve to run the tests on the created model

"""

import json
import time
import tempfile
import asyncio
import mlflow

from rasa.model_testing import test_nlu

def extract_metric(file, metric_name):
    with open(file, "r") as f:
        metrics = json.load(f)

    return metrics["weighted avg"][metric_name]


# get_diet_config = lambda config: [
#     component
#     for component in config["pipeline"]
#     if component["name"] == "DIETClassifier"][0]


import click
@click.command()
@click.option("--model_path", help="Path txt pointing to the last trained model in mlrun files")
@click.option("--validation", help="Path pointing to the validation files")
def train_and_evaluate_nlu_model(model_path, validation):
    start_time = time.time()

    with tempfile.TemporaryDirectory() as temp_results_dir:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(
            test_nlu( # faster than both compute_metrics, run_evaluation
            model=model_path,
            nlu_data=validation,
            output_directory=temp_results_dir,
            additional_arguments={},
            ))
        support = extract_metric(f"{temp_results_dir}/intent_report.json","support")
        f1_intent = extract_metric(f"{temp_results_dir}/intent_report.json","f1-score")
        f1_entity = extract_metric(f"{temp_results_dir}/DIETClassifier_report.json","f1-score")
        mlflow.log_artifacts(temp_results_dir,  artifact_path="reports")

    end_time = time.time()
    print(f"Training time: {end_time - start_time}")
    mlflow.log_metric("support", support)
    mlflow.log_metric("f1-intent", f1_intent)
    mlflow.log_metric("f1-entity", f1_entity)
    mlflow.log_metric("duration", end_time - start_time)
    return {
        "support": support,
        "f1_intent": f1_intent,
        "f1_entity": f1_entity,
        "elapsed_time": end_time - start_time,
    }

if __name__=="__main__":
    train_and_evaluate_nlu_model()
    # # for manual execution uncomment following and comment precceding
    # model_path = "mlruns/510145974657371863/4c944a69c2634bccb27cec407d2cf085/artifacts/model"
    # validation = "test_data.yml"
    # train_and_evaluate_nlu_model(model_path, validation)
