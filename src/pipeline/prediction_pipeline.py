import os, sys
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

import numpy as np
import pandas as pd

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model.pkl")
            
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            
            data_scaled = preprocessor.transform(features)
            
            pred = model.predict(data_scaled)
            return pred
        
            
        except Exception as e:
            logging.info("Exception occur in prediction")
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 fixed_acidity:float,
                 volatile_acidity:float,
                 citric_acid: float,
                 residual_sugar: float,
                 chlorides: float,
                 free_sulfur_dioxide: float,
                 sulfur_dioxide: float,
                 density: float,
                 pH: float,
                 sulphates: float,
                 alcohol:float):
        self.fixed_acidity = fixed_acidity
        self.volatile_acidity = volatile_acidity
        self.citric_acid = citric_acid
        self.residual_sugar = residual_sugar
        self.chlorides = chlorides
        self.free_sulfur_dioxide = free_sulfur_dioxide
        self.sulfur_dioxide = sulfur_dioxide
        self.density = density
        self.pH  = pH
        self.sulphates = sulphates
        self.alcohol = alcohol

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "fixed_acidity" : [self.fixed_acidity],
                "volatile_acidity" : [self.volatile_acidity],
                "citric_acid" : [self.citric_acid],
                "residual_sugar" : [self.residual_sugar],
                "chlorides" : [self.chlorides],
                "free_sulfur_dioxide" : [self.sulfur_dioxide],
                "sulfur_dioxide" : [self.sulfur_dioxide],
                "density" : [self.density],
                "pH" : [self.pH],
                "sulphates" : [self.sulphates],
                "alcohol" : [self.alcohol]
            }
            
            df = pd.DataFrame(custom_data_input_dict)
            logging.info("Dataframe is ready")
            
            return df
        except Exception as e:
            logging.info("Exception occur in prediction pipeline")
            raise CustomException(e, sys)