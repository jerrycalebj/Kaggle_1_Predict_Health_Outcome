import pandas as pd

from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.exception import CustomException
from src.logger import logging
from src.utils import *

@dataclass
class DataTransformationConfig :
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation : 
    def __init__ (self) :
        self.data_transformation_config = DataTransformationConfig()

    def get_transformer_obj(self) :
        try :
            pipeline = Pipeline(steps=[('round_columns', FunctionTransformer(round_columns_to_nearest_multiple_of_5))])

            logging.info('Reading Pipeline')

            return pipeline
        
        except Exception as e :
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)

            logging.info("Train and test data is read")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_transformer_obj()

            train_arr = preprocessing_obj.fit_transform(df_train)
            test_arr = preprocessing_obj.transform(df_test)
            
            logging.info(f"Preprocessing is complete")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj)
            logging.info(f"Preprocessing file is saved")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e :
            raise CustomException(e,sys)