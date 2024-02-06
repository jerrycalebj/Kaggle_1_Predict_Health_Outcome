import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from dataclasses import dataclass

@dataclass
class DataIngestionConig:
    train_data_path : str = os.path.join('artifacts','train.csv')
    test_data_path : str = os.path.join('artifacts','test.csv')

class DataIngestion :
    def __init__(self) :
        self.ingestion_config = DataIngestionConig()
    
    def initiate_data_ingestion(self) :
        logging.info("Data ingestion in progress")

        try : 
            df_train = pd.read_csv('Data/train.csv')
            df_test = pd.read_csv('Data/train_dataset.csv')
            logging.info('Reading the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok = True)
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path),exist_ok = True)
            
            df_train.to_csv(self.ingestion_config.train_data_path,index = False, header = True)
            df_test.to_csv(self.ingestion_config.test_data_path,index = False, header = True)
            logging.info('Data ingestion completed')

            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)

        except Exception as e :
            raise CustomException(e, sys)

if __name__ == "__main__" : 
    obj = DataIngestion()
    obj.initiate_data_ingestion()