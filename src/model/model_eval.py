import os
import numpy as np
import pandas as pd

import pickle
import json

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

#test_data = pd.read_csv('./data/processed/test_processed.csv')
def load_data(filepath: str) -> pd.DataFrame:
    try:    
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f'Error loading data from {filepath}: {e}')

#X_test = test_data.iloc[:,0:-1].valuesS
#y_test = test_data.iloc[:,-1].values

def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    try:
        X = data.drop(columns=['Potability'],axis=1)
        y = data['Potability']
        return X,y
    except Exception as e:
        raise Exception(f'Error preparing Data:{e}')
    
#model = pickle.load(open('model.pkl','rb'))

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
        with open('metrics.json','w') as file:
            json.dump(metric_dict,file,indent=4)
    except Exception as e:
        raise Exception(f'Error saving metrics to {filepath}:{e}')

def main():
    try:
        test_data_path = './data/processed/test_processed.csv'
        model_path = 'model.pkl'
        metrics_path = 'metrics.json'

        test_data = load_data(test_data_path)
        x_test, y_test = prepare_data(test_data)
        model = load_model(model_path)
        metrics = evaluation_model(model,x_test,y_test)
        save_metrics(metrics,metrics_path)
    except Exception as e:
        raise Exception(f'An error occured: {e}')
if __name__ == "__main__":
    main()
