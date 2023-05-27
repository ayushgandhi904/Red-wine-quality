import os, sys, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            #Model Training
            model.fit(X_train, y_train)
            
            #Predicting Training data
            # y_train_pred = model.predict(X_train)
            
            #Predicting Testing data
            y_pred = model.predict(X_test)
            
            #Getting accuracy score
            
            # train_model_score = accuracy_score(y_train, y_train_pred)
            test_model_score = accuracy_score(y_test, y_pred)
            
            report[list(models.keys)()[i]] = test_model_score
            
        return report
    
    except Exception as e:
        logging.info("Error in evaluate model")
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:

        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        logging.info("Exception occured in load_object function utils")
        raise CustomException(e, sys)
        