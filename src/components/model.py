import os
import sys
from dataclasses import dataclass

from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelConfig :
    model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer :
    def __init__(self) :
        self.model_file = ModelConfig()

    def initiate_model(self, train_array, test_array) : 
        try : 
            logging.info("Splitting train and test data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1], train_array[:,-1],
                test_array[:,:-1], test_array[:,-1]
            )

            models = {
                "XGBClassifier": XGBClassifier(),
                "CatBoostClassifier": CatBoostClassifier()
                }
            
            params = {
                
                "XGBClassifier":{
                    'learning_rate':[.1,.01],
                    'n_estimators': [1500,2000,3000]
                },
                "CatBoostClassifier":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [50, 100]
                }
                
            }
            models_report : dict = evaluate_models(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, models = models, param = params)

            best_model_score = max(sorted(models_report.values()))

            best_model_name = list(models_report.keys())[
                list(models_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.7 :
                raise CustomException("Low accuracy score is observed")
            logging.info(f"Best found model is {best_model} with accuracy of {best_model_score}")

            save_object(
                file_path = self.model_file.model_file_path,
                obj = best_model
            )

            return best_model_score

        except Exception as e :
            raise CustomException(e,sys)