# MLFLOW Hyper paramater optimisation for Rasa

This project aims to provide an example on how to use MLFLOW within rasa in order to track `config.yaml` updates impact on the bot performance.

It serves to automatically track multiple experiments of rasa train on a defined search space using the hyperopt methodology and identifies the best combination.

## Installation

First you need to create and setup a virtual environment to install dependencies needed to run the code by running within your venv the following commands in the project root directory.

```
python3.8 -m venv .mlflow 
source .mlflow/bin/activate
pip install --upgrade pip
pip install mlflow
```

Second, you need to update the files of the sub-project you want to execute by adding your own `test_data.yml `and `training_data.yml` and `config.yml` files.

You can generate a test/train split in rasa using the command `rasa data split nlu` from you rasa project root directory. Than copy/paste them in files.

`config.yml` should be added as a template as in the example, where you replace the values with the their name.

## Usage

The code is organized as a `MLFLOW Project`. It contains a complete workflow that spans a search space in search for best parameters and leverages multithreading for faster execution.

You can use this project in different ways:

- By running from root directory: `mlflow run. <pass you own params if needed>` .
- If you don't want to clone this project, you can run this command :
  - without default params: `mlflow run git@github.com:YahyaGrb/mlflow_rasa_hp.git`
  - with your own params/files: `mlflow run git@github.com:YahyaGrb/mlflow_rasa_hp.git -P <your params if needed>`
- From a jupyter notebook:
  - ```
    import mlflow
    project_uri = "https://github.com/YahyaGrb/mlops_rasa/mlflow_rasa_hp"
    params = your_params # (as needed by the project)

    mlflow.run(project_uri, parameter=params)
    ```

## Results

You can track the results in Mlflow by running `mlflow ui` and opening `http://127.0.0.1:5000` from your browser.
