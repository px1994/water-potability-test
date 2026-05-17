import os
import numpy as np
import pandas as pd

import pickle
import json
import mlflow

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

from mlflow import log_metric, log_param, log_artifact
import mlflow.sklearn
import dagshub
from mlflow.models import infer_signature


# =========================================================
# DAGSHUB TOKEN AUTHENTICATION
# =========================================================

DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

if not DAGSHUB_TOKEN:
    raise EnvironmentError("DAGSHUB_TOKEN Environment variable is not set")

os.environ["DAGSHUB_USER_TOKEN"] = DAGSHUB_TOKEN

# MLflow authentication
os.environ["MLFLOW_TRACKING_USERNAME"] = "pritesh13590"
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

# =========================================================
# INITIALIZE DAGSHUB + MLFLOW
# =========================================================

dagshub.init(
    repo_owner='pritesh13590',
    repo_name='water-potability-test',
    mlflow=True
)

mlflow.set_tracking_uri("https://dagshub.com/pritesh13590/water-potability-test.mlflow")



# Set the experiment name in MLflow

mlflow.set_experiment("water-potability-exp")



def load_data(filepath: str) -> pd.DataFrame:
    try:    
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f'Error loading data from {filepath}: {e}')



def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    try:
        X = data.drop(columns=['Potability'],axis=1)
        y = data['Potability']
        return X,y
    except Exception as e:
        raise Exception(f'Error preparing Data:{e}')
    
# load model path

def load_model(filepath: str):
    try:
        with open(filepath,'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        raise Exception(f'Error loading model from {filepath} :{e}')

def evaluation_model(model, x_test:pd.DataFrame, y_test:pd.Series) -> dict:
    try:
        y_pred = model.predict(x_test)

        metric_dict = {

            'accuracy': accuracy_score(y_test,y_pred),
            'precision': precision_score(y_test,y_pred),
            'recall': recall_score(y_test,y_pred),
            'f1score': f1_score(y_test,y_pred)
        }
        return metric_dict
    except Exception as e:
        raise Exception(f'Error evaluating model: {e}')
    
def save_metrics(metric_dict, filepath:str) -> None:
    try:
        with open(filepath, 'w') as file:
            json.dump(metric_dict, file, indent=4)
    except Exception as e:
        raise Exception(f'Error saving metrics to {filepath}: {e}')

def main():
    try:
        test_data_path = './data/processed/test_processed.csv'
        model_path = 'models/model.pkl'
        metrics_path = 'reports/metrics.json'
        model_name = 'Best_Model'
        
        test_data = load_data(test_data_path)
        x_test, y_test = prepare_data(test_data)
        model = load_model(model_path)

        # Start MLflow run
        with mlflow.start_run() as run:

            metrics = evaluation_model(model, x_test, y_test)

            save_metrics(metrics, metrics_path)

            # log metrics
            mlflow.log_metrics(metrics)

            # log artifacts
            mlflow.log_artifact(model_path)
            mlflow.log_artifact(metrics_path)
            mlflow.log_artifact(__file__)

            # model signature
            signature = infer_signature(
                x_test,
                model.predict(x_test)
            )

            # log model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="Best_Model",
                signature=signature
            )

            # save run info
            run_info = {
                'run_id': run.info.run_id,
                'model_name': "Best_Model"
            }

            with open("reports/run_info.json", 'w') as file:
                json.dump(run_info, file, indent=4)


    except Exception as e:
        raise Exception(f'An error occured: {e}')
    
if __name__ == "__main__":
    main()
