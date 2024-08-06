from dataclasses import dataclass
import numpy as np
import pandas as pd
import os
import sys
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import lg
from src.utils import save_object

@dataclass
class DataTranformationConfig:
    preProcessorObjFile_path = os.path.join('arifacts','proprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTranformationConfig()
        
    def get_data_tranform_object(self):
        try:
            numerical_Columns = ["writing_score","reading_score"]
            categorical_columns = ["gender","race_ethnicity","parental_level_of_education",
                                   "lunch","test_preparation_course"]
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            lg.info("Categorical columns standard scaling completed")
            
            lg.info("Numericalw columns standard scaling completed")
            
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_Columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )
            return  preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
    
    def init_ate_data_transform(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            lg.info("Reading files")
            
            preprocessingObj = self.get_data_tranform_object()
            target_column_name = "math_score"
            
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            lg.info("Applying preprocessisng and tranformation on data")
            
            input_feature_train_arr = preprocessingObj.fit_transform(input_feature_train_df)
            input_feature_test_arr= preprocessingObj.transform(input_feature_test_df)
            
            train_arr = np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            
            test_arr = np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]
            lg.info(f"Saved preprocessing object.")
            
            save_object(
                file_path = self.data_transformation_config.preProcessorObjFile_path,
                obj=preprocessingObj
            )
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preProcessorObjFile_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)
            
            