import os,sys
from src.logger import logging
from src.exception import CustomException
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
#Intialize data ingestion   

@dataclass
class DataIngestionconfig:
    train_data_path:str = os.path.join("artifacts", "train.csv")
    test_data_path:str = os.path.join("artifacts", "test.csv")
    raw_data_path:str = os.path.join("artifacts", "raw.csv")
    

#Data Ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionconfig()
        
    def initiate_data_ingestion(self):
        logging.info("Data ingestion method started")
        try:
            df = pd.read_csv(os.path.join("notebooks/data","red_wine_quality.csv"))
            df['quality'] = df['quality'].apply(lambda x: 1 if x > 5.5 else 0)
            df.drop_duplicates(inplace=True)
            logging.info("Dataset readed by Pandas")
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok= True)
            df.to_csv(self.ingestion_config.raw_data_path, index = False)
            logging.info("Train test split")
            train_set, test_set = train_test_split(df, test_size = 0.3, random_state=100)
            
            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)
            logging.info("Data Ingestion completed")
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )                                       
        except Exception as e:
            logging.info("Exception occur as Data Ingestion stage")
            raise CustomException(e, sys)
