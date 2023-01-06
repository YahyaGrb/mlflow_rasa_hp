"""
This script will serve to run the tests on the created model

"""

import json
import time
import tempfile
import asyncio
import mlflow
from rasa.core.processor import MessageProcessor
from rasa.core.agent import Agent
from rasa.nlu.test import compute_metrics, run_evaluation
from rasa.shared.nlu.training_data.loading import load_data
from rasa.model_testing import test_nlu

def extract_metric(file, metric_name):
    with open(file, "r") as f:
        metrics = json.load(f)

    return metrics["weighted avg"][metric_name]


# get_diet_config = lambda config: [
#     component
#     for component in config["pipeline"]
#     if component["name"] == "DIETClassifier"][0]


# import Path

import click
import os
# @click.command()
# @click.option("--model_path", help="Path readable by Spark to the ratings Parquet file")
# @click.option("--validation", help="Path readable by Spark to the ratings Parquet file")
def train_and_evaluate_nlu_model(model_path, validation):
    print("startring evooos")
    print(model_path)
    start_time = time.time()
    agent = Agent.load(model_path)
    message_processor = MessageProcessor(
        model_path=model_path,
        tracker_store=agent.tracker_store,
        lock_store=agent.lock_store,
        generator=agent.nlg)
    data = load_data(validation)
    print("prep time for loop 1 and 2", time.time() - start_time)
    print("running loop1")
    start_time = time.time()
    loop1 = asyncio.new_event_loop()
    asyncio.set_event_loop(loop1)
    metrics = loop1.run_until_complete(compute_metrics(message_processor,data))
    loop1.close()
    print(f"duration 1= ", start_time - time.time() )
    print("running loop2")
    start_time = time.time()

    loop2 = asyncio.new_event_loop()
    asyncio.set_event_loop(loop2)
    metrics = loop2.run_until_complete(run_evaluation(validation, message_processor, output_directory="reslts"))
    print(f"duration 2= ", start_time - time.time() )
    
    print("running loop3")
    start_time = time.time()

    with tempfile.TemporaryDirectory() as temp_results_dir:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        res = loop.run_until_complete(
            test_nlu(
            model=model_path,
            nlu_data=validation,
            output_directory=temp_results_dir,
            additional_arguments={},
            ))
        print(f"duration 3= ", start_time - time.time() )
        
        support = extract_metric(f"{temp_results_dir}/intent_report.json","support")
        print("supp: ", support)
        f1_intent = extract_metric(f"{temp_results_dir}/intent_report.json","f1-score")
        f1_entity = extract_metric(f"{temp_results_dir}/DIETClassifier_report.json","f1-score")
        mlflow.log_artifacts(temp_results_dir,  artifact_path="reports")

    end_time = time.time()
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
    model_path = "/mlruns/0/e59dfe032aec460c9af705de5ad53e6a/artifacts/model/20221222-175225-square-pique.tar.gz"
    validation = "./test_data.yml"
    train_and_evaluate_nlu_model(model_path, validation)
