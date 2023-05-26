import os, sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_model
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting Dependent & Independent variable from train & test data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            logging.info("Data splitted")
            
            models = {
                "LogisticRegression" : LogisticRegression(),
                "RidgeClassifier" : RidgeClassifier(),
                "LogisticRegressionCV" : LogisticRegressionCV(),
                "DecisionTree" : DecisionTreeClassifier(),
                "SVC" : SVC(),
                "RandomForest" : RandomForestClassifier(),
                "GradientClassifier" : GradientBoostingClassifier()
            }
            model_report:dict = evaluate_model(X_train, y_train, X_test, y_test, models)    
            print(model_report)
            print("\n******************")
            logging.info(f"Model Report of :{model_report}")
            
            #To get best model score
            
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            print(f"Best Model {best_model_name}, accuracy score :{best_model_name}")
            print("\n***********")
            logging.info(f"Best Model {best_model_name}, accuracy score :{best_model_name}")
            
            save_object(
                
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            
        except Exception as e:
            logging.info("Exception occur at Model Traiing")
            raise CustomException(e, sys)
            