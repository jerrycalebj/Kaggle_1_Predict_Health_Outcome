import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")

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


def feature_engineering1(df):
    df_func = df.copy()
    df_func['BMI'] = df_func['weight(kg)'] / ((df_func['height(cm)'] / 100) ** 2)
    bins = [0, 18.5, 24.9, 29.9,df_func['BMI'].max()]
    labels = [0, 1, 2, 3]  
    df_func['BMI bins'] = pd.cut(df_func['BMI'], bins=bins, labels=labels, include_lowest=True)
    df_func['Ratio Lipids'] = df_func['triglyceride'] / df_func['HDL'] 
    df_func['Height*Hemoglobin'] = df_func['height(cm)'] * df_func['hemoglobin']
    df_func['HDL * Chol'] = df_func['HDL'] * df_func['Cholesterol']
    df_func['HTG'] = df_func['height(cm)'] * df_func['triglyceride'] * df_func['Gtp']
    df_func['Combined'] = df_func['height(cm)'] * df_func['triglyceride'] * df_func['Gtp'] * df_func['hemoglobin'] * df_func['ALT'] * df_func['weight(kg)'] * df_func['serum creatinine'] / 10000
    df_func['hearing'] = df_func.apply(lambda row: 1 if (row['hearing(right)'] == 2 or row['hearing(left)'] == 2) else 0, axis=1)
    return df_func

def dimensionality_reduction(df):
    df_copy = df.copy()
    df_copy.loc[df_copy['age'] > 75, 'age'] = 75
    df_copy.loc[df_copy['height(cm)'] < 145, 'height(cm)'] = 145
    df_copy.loc[df_copy['height(cm)'] > 185, 'height(cm)'] = 185
    df_copy.loc[df_copy['weight(kg)'] < 45, 'weight(kg)'] = 45     
    df_copy.loc[df_copy['weight(kg)'] > 100, 'weight(kg)'] = 100 
    df_copy.loc[df_copy['Urine protein'] > 2, 'Urine protein'] = 2
    df_copy.loc[df_copy['eyesight(left)'] > 2, 'eyesight(left)'] = 2.1
    df_copy.loc[df_copy['eyesight(right)'] > 2, 'eyesight(right)'] = 2.1
    df_copy.loc[df_copy['serum creatinine'] > 1.5, 'serum creatinine'] = 1.6
    df_copy.loc[df_copy['serum creatinine'] < 0.3, 'serum creatinine'] = 0.3

    return df_copy


lower_range1 = [300,350,300,120,200,350,10,1.5,1.5,115,35,28000,3250,7000000,50000000]
upper_range1 = [310,360,310,140,210,400,15,1.75,1.75,120,37,30000,3300,8000000,100000000]
cols_to_modify1 = ['fasting blood sugar','Cholesterol','LDL','AST','ALT','Gtp','Ratio Lipids','eyesight(left)','eyesight(right)','waist(cm)','BMI','HDL * Chol','Height*Hemoglobin','HTG','Combined']

def outlier_reduction1(lower_range1,upper_range1,cols_to_modify1,df):
    df_copy = df.copy()
    for col,lower,upper in zip(cols_to_modify1,lower_range1,upper_range1):
        mask = df_copy[col] > lower
        if mask.any():
            #MinMax scaler need 2D data
            values_to_scale = df_copy.loc[mask, col].values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(lower, upper))
            df_copy.loc[mask, col] = scaler.fit_transform(values_to_scale)
    return df_copy

lower_range2 = [180,110,200,290,400,140,230,120,160,250,9,1,1,24000,20000000]
upper_range2 = [190,120,230,320,410,150,250,140,190,300,11,1.4,1.4,25000,30000000]
cols_to_modify2 = ['systolic','relaxation','fasting blood sugar','Cholesterol','triglyceride','HDL','LDL','AST','ALT','Gtp','Ratio Lipids','eyesight(left)','eyesight(right)','HDL * Chol','Combined']
  
def outlier_reduction2(lower_range2,upper_range2,cols_to_modify2,df):
    df_copy = df.copy()
    for col,lower,upper in zip(cols_to_modify2,lower_range2,upper_range2):
        mask = df_copy[col] > lower
        if mask.any():
            #MinMax scaler need 2D data
            values_to_scale = df_copy.loc[mask, col].values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(lower, upper))
            df_copy.loc[mask, col] = scaler.fit_transform(values_to_scale)
    return df_copy

lower_range3 = [15,7.5,70,55,20,4000,1400]
upper_range3 = [20,8,100,65,25,5000,1500]
cols_to_modify3 = ['HDL','hemoglobin','Cholesterol','fasting blood sugar','LDL','HDL * Chol','Height*Hemoglobin']

def outlier_reduction3(lower_range3,upper_range3,cols_to_modify3,df):
    df_copy = df.copy()
    for col,lower,upper in zip(cols_to_modify3,lower_range3,upper_range3):
        mask = df_copy[col] < upper
        if mask.any():
            #MinMax scaler need 2D data
            values_to_scale = df_copy.loc[mask, col].values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(lower, upper))
            df_copy.loc[mask, col] = scaler.fit_transform(values_to_scale)
    return df_copy

lower_range4 = [80,120,150,200,105,110,210]
upper_range4 = [100,140,175,250,110,120,220]
cols_to_modify4 = ['AST','ALT','fasting blood sugar','Gtp','relaxation','HDL','LDL']
  
def outlier_reduction4(lower_range4,upper_range4,cols_to_modify4,df):
    df_copy = df.copy()
    for col,lower,upper in zip(cols_to_modify4,lower_range4,upper_range4):
        mask = df_copy[col] > lower
        if mask.any():
            #MinMax scaler need 2D data
            values_to_scale = df_copy.loc[mask, col].values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(lower, upper))
            df_copy.loc[mask, col] = scaler.fit_transform(values_to_scale)
    return df_copy

lower_range5 = [60,80,140,140,10000000]
upper_range5 = [70,90,150,150,15000000]
cols_to_modify5 = ['AST','ALT','Gtp','fasting blood sugar','Combined']
  
def outlier_reduction5(lower_range5,upper_range5,cols_to_modify5,df):
    df_copy = df.copy()
    for col,lower,upper in zip(cols_to_modify5,lower_range5,upper_range5):
        mask = df_copy[col] > lower
        if mask.any():
            #MinMax scaler need 2D data
            values_to_scale = df_copy.loc[mask, col].values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(lower, upper))
            df_copy.loc[mask, col] = scaler.fit_transform(values_to_scale)
    return df_copy

lower_range6 = [50,70,100,5000000]
upper_range6 = [60,80,120,7000000]
cols_to_modify6 = ['AST','ALT','Gtp','Combined']
  
def outlier_reduction6(lower_range6,upper_range6,cols_to_modify6,df):
    df_copy = df.copy()
    for col,lower,upper in zip(cols_to_modify6,lower_range6,upper_range6):
        mask = df_copy[col] > lower
        if mask.any():
            #MinMax scaler need 2D data
            values_to_scale = df_copy.loc[mask, col].values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(lower, upper))
            df_copy.loc[mask, col] = scaler.fit_transform(values_to_scale)
    return df_copy

cols_to_encode = ['dental caries', 'BMI bins']

def one_hot_encodefunc(df, cols_to_encode):
    df_func = df.copy()
    df_encoded = pd.get_dummies(df_func, columns = cols_to_encode, drop_first=True)
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'bool':
            df_encoded[col] = df_encoded[col].astype(int)
    return df_encoded

def feature_removal(df):
    df_func = df.copy()
    df_func = df_func.drop('hearing(left)', axis=1)
    df_func = df_func.drop('hearing(right)', axis=1)
    df_func = df_func.drop('Urine protein', axis=1)
    df_func = df_func.drop('BMI bins_3', axis=1)
    df_func = df_func.drop('Combined', axis=1)
    df_func = df_func.drop('HTG', axis=1)
    df_func = df_func.drop('HDL * Chol', axis=1)
    df_func = df_func.drop('Ratio Lipids', axis=1)
    return df_func

def process_data(df):
    df_func = df.copy()
    df_func = round_columns_to_nearest_multiple_of_5(df_func)
    df_func = feature_engineering1(df_func)
    df_func = dimensionality_reduction(df_func)
    df_func = outlier_reduction1(lower_range1, upper_range1, cols_to_modify1, df_func)
    df_func = outlier_reduction2(lower_range2, upper_range2, cols_to_modify2, df_func)
    df_func = outlier_reduction3(lower_range3, upper_range3, cols_to_modify3, df_func)
    df_func = outlier_reduction4(lower_range4, upper_range4, cols_to_modify4, df_func)
    df_func = outlier_reduction5(lower_range5, upper_range5, cols_to_modify5, df_func)
    df_func = outlier_reduction6(lower_range6, upper_range6, cols_to_modify6, df_func)
    df_func = one_hot_encodefunc(df_func, cols_to_encode)
    df_func = feature_removal(df_func)
    return df_func


def data_preprocessing(temp):
    train_to_scale = ['age', 'height(cm)', 'weight(kg)', 'waist(cm)', 'eyesight(left)',
       'eyesight(right)', 'systolic', 'relaxation', 'fasting blood sugar',
       'Cholesterol', 'HDL', 'LDL', 'hemoglobin',
       'serum creatinine', 'BMI', 'Height*Hemoglobin']
    cols_to_trans = ['AST', 'ALT', 'Gtp' , 'triglyceride']
    sc = RobustScaler()
    scaled_data = sc.fit_transform(temp[train_to_scale])
    scaled_train = pd.DataFrame(scaled_data, columns=train_to_scale)
    
    unscaled_columns = temp.drop(train_to_scale, axis=1)  
    unscaled_columns = unscaled_columns.drop(cols_to_trans, axis=1)  
    transformed_column = np.sqrt(temp[cols_to_trans])
    
    unscaled_columns.index = scaled_train.index
    transformed_column.index = scaled_train.index
    
    X = pd.concat([scaled_train, unscaled_columns], axis=1)
    X = pd.concat([X, transformed_column], axis=1)
    
    return X

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_accuracy

        return report

    except Exception as e:
        raise CustomException(e, sys)
    