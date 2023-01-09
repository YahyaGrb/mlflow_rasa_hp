# MLFLOW tracking and Hyper paramater optimisation for Rasa

This project aims to provide an example on how to use MLFLOW within rasa in order to track `config.yaml` updates impact on the bot performance.

The project is organized in two parts totally independent from each other.

The `Track_rasa` serves to track a rasa train performance for a single config file.

The `Hyperopt_rasa` serves to automatically track multiple experiements of rasa train on a defined search space using the hyperopt methodology and identifies the best combination.

## Installation

First you need to create and setup a virtual environment to install dependencies needed to run the code by running within your venv the following commands in the project root directory.

```
python3.8 -m venv .mlflow 
source .mlflow/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Second, you need to update the files of the sub-project you want to exeucte by adding you own `test_data.yml `and `training_data.yml` and `config.yml` files.

You can generate a test/train split in rasa using the command `rasa data split nlu` from you rasa project root directory. Than copy/paste them in files.

`config.yml` should be added as a template as in the example, where you replace the values with the their name.

You need to load `Spacy` model if you config uses spacy.

## Usage

### Track_rasa

Track is a simple workflow that trains and tests rasa using a MLFlow project workflow method.

At the root directory, run `mlflow run track_rasa --env-manager=local <pass you own params if needed>`

You can track the results in Mlflow running `mlflow ui` and opening `http://127.0.0.1:5000` from you browser.

### Hyperopt_rasa

`Hyperopt_rasa` is a more complete workflow that spans a search space in search for best parameters. It uses Mlflow prallelism and multithreading for faster execution.

Same as for `track_rasa`, you can use this project running `mlflow run hyperopt_rasa --env-manager=local <pass you own params if needed>` from root directory.

You can track the results in Mlflow running `mlflow ui` and opening `http://127.0.0.1:5000` from you browser.

### Github project usage (WIP)

you can use this project standalone withou cloning and so by leveraging the MLProjects feature

all you need is

but i need to divi it into two different projects so that will work, or add a param to launch either one or the other (nt sure this will work))

```
import mlflow
project_uri = "https://github.com/YahyaGrb/mlops_rasa/track_rasa" # or hyperopt_rasa
params = your_params # (as accepted by the project)

mlflow.run(project_uri, parameter=params)

```
