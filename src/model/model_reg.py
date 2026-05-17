# =========================================================
# IMPORT REQUIRED LIBRARIES
# =========================================================

import json
import os

# MLflow
import mlflow
from mlflow.tracking import MlflowClient

# DagsHub
import dagshub

# =========================================================
# INITIALIZE DAGSHUB + MLFLOW
# =========================================================

dagshub.init(
    repo_owner='pritesh13590',
    repo_name='water-potability',
    mlflow=True
)

# Set tracking URI
mlflow.set_tracking_uri(
    "https://dagshub.com/pritesh13590/water-potability.mlflow"
)

# Set credentials
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

# =========================================================
# SET EXPERIMENT
# =========================================================

mlflow.set_experiment("DVC PIPELINE")

# =========================================================
# LOAD RUN INFO
# =========================================================

reports_path = "reports/run_info.json"

try:
    with open(reports_path, "r") as file:
        run_info = json.load(file)

    run_id = run_info["run_id"]
    model_name = run_info["model_name"]

    print(f"Run ID: {run_id}")
    print(f"Model Name: {model_name}")

except Exception as e:
    raise Exception(f"Error loading run info: {e}")

# =========================================================
# CREATE MLFLOW CLIENT
# =========================================================

client = MlflowClient()

# =========================================================
# VERIFY ARTIFACT EXISTS
# =========================================================

try:
    artifacts = client.list_artifacts(run_id)

    artifact_paths = [artifact.path for artifact in artifacts]

    print("\nAvailable Artifacts:")
    print(artifact_paths)

    if model_name not in artifact_paths:
        raise Exception(
            f"Model artifact '{model_name}' not found "
            f"inside run '{run_id}'"
        )

except Exception as e:
    raise Exception(f"Error checking artifacts: {e}")

# =========================================================
# CREATE MODEL URI
# =========================================================

# IMPORTANT:
# Do NOT add 'artifacts/' manually

model_uri = f"runs:/{run_id}/{model_name}"

print(f"\nModel URI: {model_uri}")

# =========================================================
# REGISTER MODEL
# =========================================================

try:
    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )

    model_version = registered_model.version

    print(
        f"\nModel Registered Successfully!"
        f"\nModel Name: {model_name}"
        f"\nVersion: {model_version}"
    )

except Exception as e:
    raise Exception(f"Error registering model: {e}")

# =========================================================
# TRANSITION MODEL TO STAGING
# =========================================================

new_stage = "Staging"

try:
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage=new_stage,
        archive_existing_versions=True
    )

    print(
        f"\nModel '{model_name}' "
        f"version {model_version} "
        f"transitioned to '{new_stage}' stage."
    )

except Exception as e:
    raise Exception(f"Error transitioning model stage: {e}")