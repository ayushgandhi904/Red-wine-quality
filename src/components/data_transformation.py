import os, sys
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.utils import save_object
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifcats", "preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformation initiated")
            num_cols = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]
            target_cols = ["quality"]  
            
            #Numerical Pipeline
            
            logging.info("Pipeline initiated")
            
            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                    
                ]
            ) 
            
            #Label pipeline 
            lab_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy = "most_frequent")),
                    ("labelencoder", LabelEncoder())
                ]
            ) 

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, num_cols),
                ("lab_pipeline", lab_pipeline, target_cols)
            ])
            
            return preprocessor
            
            logging.info("Pipeline completed")
            
            
        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e, sys)
    
    def initaite_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Train & Test data readed")
            logging.info(f"Train head : \n{train_df.head().to_string()}")
            logging.info(f"Test head : \n{test_df.head().to_string()}")
            
            logging.info("Obtaining preprocessing object")
            
            preprocessing_obj = self.get_data_transformation_object()
            
            target_column = "quality"
            drop_columns = [target_column]
            
            input_feature_train_df = train_df.drop(columns = drop_columns, axis = 1)
            target_featrue_train_df = train_df[target_column]
            
            input_feature_test_df = test_df.drop(columns = drop_columns, axis = 1)
            target_feature_test_df = test_df[target_column]
            
            #Transforming into preprocessor
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            logging.info("Applied preprocessing to training & test datasets")
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_featrue_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            save_object(
                
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            logging.info("Preprocessor pickle file saved")
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            logging.info("Exception occur in initiate_datatransformation")
            raise CustomException(e, sys)