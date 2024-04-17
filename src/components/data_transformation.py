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
    train_arr_path : str = os.path.join('artifacts','train_arr.csv')
    test_arr_path : str = os.path.join('artifacts','test_arr.csv')


class DataTransformation : 
    def __init__ (self) :
        self.data_transformation_config = DataTransformationConfig()

    def get_preprocess_obj(self) :
        try :
            pipeline = Pipeline(steps=[('data_preprocessing', FunctionTransformer(process_data)),
                                       ('data_transformer',FunctionTransformer(data_preprocessing))])
            logging.info('Reading Data Preprocessing and Processing Pipeline')
            return pipeline
        
        except Exception as e :
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self,train_path,test_path):

        try:
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)

            logging.info("Train and test data is read")
            logging.info("Obtaining preprocessing object")

            target_column_name="smoking"
            input_feature_train_df = df_train.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = df_train[target_column_name]

            input_feature_test_df = df_test.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = df_test[target_column_name]

            preprocessing_obj = self.get_preprocess_obj()
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info(f"Preprocessing is complete")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj)
            logging.info(f"Preprocessing file is saved")
            
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            train_df = pd.DataFrame(train_arr)
            test_df = pd.DataFrame(test_arr)

            # Save DataFrames as CSV files
            train_df.to_csv(self.data_transformation_config.train_arr_path, index=False, header=False)
            test_df.to_csv(self.data_transformation_config.test_arr_path, index=False, header=False)

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e :
            raise CustomException(e,sys)