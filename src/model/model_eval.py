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
import mlflow
from mlflow.models import infer_signature


# INITIALIZE DAGSHUB + MLFLOW
dagshub.init(repo_owner='pritesh13590', 
             repo_name='water-potability', 
             mlflow=True)

# Set MLflow tracking URI
mlflow.set_tracking_uri("https://dagshub.com/pritesh13590/water-potability.mlflow")


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
    


def load_model(filepath: str):
    try:
        with open(filepath,'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        raise Exception(f'Error loading model from {filepath} :{e}')

def evaluation_model(model, x_test:pd.DataFrame, y_test:pd.Series, model_name: str) -> dict:
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
        with open('reports/metrics.json','w') as file:
            json.dump(metric_dict,file,indent=4)
    except Exception as e:
        raise Exception(f'Error saving metrics to {filepath}:{e}')

def main():
    try:
        test_data_path = './data/processed/test_processed.csv'
        model_path = 'models/model.pkl'
        metrics_path = 'reports/metrics.json'
        model_name = 'Best_Model'
        
        test_data = load_data(test_data_path)
        x_test, y_test = prepare_data(test_data)
        model = load_model(model_path)
        metrics = evaluation_model(model,x_test,y_test, model_name)
        save_metrics(metrics,metrics_path)

        # Start MLflow run
        with mlflow.start_run() as run:
            metrics = evaluation_model(model, x_test, y_test, model_name)
            save_metrics(metrics, metrics_path)

            # Log artifacts
            mlflow.log_artifact(model_path)
            mlflow.log_artifact(metrics_path)
            
            # Log the source code file
            mlflow.log_artifact(__file__)

            signature = infer_signature(x_test,model.predict(x_test))

            mlflow.sklearn.log_model(model,"Best Model",signature=signature)

            #Save run ID and model info to JSON File
            run_info = {'run_id': run.info.run_id, 'model_name': "Best Model"}
            reports_path = "reports/run_info.json"
            with open(reports_path, 'w') as file:
                json.dump(run_info, file, indent=4)



    except Exception as e:
        raise Exception(f'An error occured: {e}')
    
if __name__ == "__main__":
    main()
