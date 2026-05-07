# =========================================================
# IMPORT REQUIRED LIBRARIES
# =========================================================

import json

# MLflow
import mlflow
from mlflow.tracking import MlflowClient

# DagsHub
import dagshub

# =========================================================
# INITIALIZE DAGSHUB + MLFLOW
# =========================================================

dagshub.init(repo_owner='pritesh13590', 
             repo_name='water-potability', 
             mlflow=True)

# Set MLflow tracking URI
mlflow.set_tracking_uri("https://dagshub.com/pritesh13590/water-potability.mlflow")


# Set MLflow Experiment
mlflow.set_experiment("Final_Model")

# =========================================================
# LOAD RUN INFORMATION
# =========================================================

reports_path = "reports/run_info.json"

with open(reports_path, "r") as file:

    run_info = json.load(file)


# Extract details
run_id = run_info.get("run_id")

model_name = run_info.get("model_name")

print(run_info)

print("Run ID     :", run_id)

print("Model Name :", model_name)

# =========================================================
# CREATE MLFLOW CLIENT
# =========================================================

client = MlflowClient()

# =========================================================
# CREATE MODEL URI
# =========================================================

model_uri = f"runs:/{run_id}/artifacts/{model_name}"

print("Model URI:", model_uri)

# =========================================================
# REGISTER MODEL
# =========================================================

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=model_name
)

# =========================================================
# GET MODEL VERSION
# =========================================================

model_version = registered_model.version

print("Registered Model Version:", model_version)

# =========================================================
# TRANSITION MODEL STAGE
# =========================================================

new_stage = "Staging"

client.transition_model_version_stage(

    name=model_name,

    version=model_version,

    stage=new_stage,

    archive_existing_versions=True
)

# =========================================================
# FINAL MESSAGE
# =========================================================

print(
    f"\nModel '{model_name}' "
    f"Version {model_version} "
    f"transitioned to '{new_stage}' stage successfully."
)