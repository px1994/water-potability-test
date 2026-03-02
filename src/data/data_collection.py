import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import yaml 

def load_params(filepath: str) -> float:
    try:
        with open(filepath,'r') as file:
            params = yaml.safe_load(file)
            test_size = params['data_collection']['test_size']
            seed = params['data_collection']['seed']
        return test_size, seed
    except Exception as e:
        raise Exception(f'Error loading parameter from {filepath}:{e}')


def load_data(filepath: str) -> pd.DataFrame:
    try: 
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f'Error loading Data from {filepath}:{e}')



def split_data(data: pd.DataFrame,test_size: float, seed: int) -> tuple[pd.DataFrame,pd.DataFrame]:
    try:
        return train_test_split(data,test_size=test_size,random_state=seed)
    except ValueError as e:
        raise ValueError(f'Error splitting data {e}')
    



def save_data(df: pd.DataFrame, filepath: str) -> None:
    try:
        df.to_csv(filepath, index=False)
    except Exception as e:
        raise Exception(f'Error saving data to {filepath}: {e}')
    


def main():   
    data_filepath = r'c:/Data/project_datasets/water_potability.csv' # data path 
    params_filepath = 'params.yaml' # params file location 
    raw_data_path = os.path.join('data','raw') # save data location
    try:
        data = load_data(data_filepath)
        test_size, seed = load_params(params_filepath)
        train_data,test_data = split_data(data, test_size, seed)
        
        os.makedirs(raw_data_path)

        save_data(train_data, os.path.join(raw_data_path,'train.csv'))
        save_data(test_data, os.path.join(raw_data_path,'test.csv'))
    except Exception as e:
        raise Exception(f'An Error Occured:{e}') 



if __name__ == '__main__':
    main()



