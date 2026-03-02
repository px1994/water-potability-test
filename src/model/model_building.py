import os
import numpy as np
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier

import yaml

def load_param(params_path:str) -> int:
    try:
        with open(params_path,'r') as file:
            params = yaml.safe_load(file)
        return params['model_building']['n_estimators']
    except Exception as e:
        raise Exception()



def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f'Error loading data from {filepath}:{e}')
    



def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame,pd.Series]:
    try:
        X = data.drop(columns=['Potability'])
        y = data['Potability']
        return X,y
    except Exception as e:
        raise Exception(f'')




def train_model(X: pd.DataFrame, y: pd.Series, n_estimator: int) -> RandomForestClassifier:
    try: 
        clf = RandomForestClassifier(n_estimators=n_estimator)
        clf.fit(X,y)
        return clf
    except Exception as e:
        raise Exception(f'Error Training Model:{e}')
    



def save_model(model: RandomForestClassifier, model_name: str) -> None:
    try:
        with open(model_name,'wb') as file:
            pickle.dump(model, file)
    except Exception as e:
        raise Exception(f'Error saving model{model_name}:{e}')



def main():
    try:
        params_path  = 'params.yaml'
        data_path = './data/processed/train_processed.csv' # file path from data processed 
        model_path = 'models/model.pkl'

        n_estimators = load_param(params_path)
        train_data = load_data(data_path)
        X_train, y_train = prepare_data(train_data)

        model = train_model(X_train, y_train, n_estimators)
        save_model(model, model_path)
    except Exception as e:
        raise Exception(f'An Error Occured {e}')


if __name__=='__main__':
    main()


