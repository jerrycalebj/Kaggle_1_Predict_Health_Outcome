import os
import sys
import pickle

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

#Preprocessing Functions
    
def round_columns_to_nearest_multiple_of_5(df):
    new_df = df.copy()
    cols_to_round = ['age','height(cm)', 'weight(kg)']
    for col in cols_to_round:
        new_df[col] = df[col].apply(lambda x: round(x / 5) * 5) 
    return new_df