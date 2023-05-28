import os
import sys

from dataclasses import dataclass
import tarfile
from six.moves import urllib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException 
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

# single compressed file, housing.tgz, which contains a
# comma-separated value (CSV) file called housing.csv with all the data.

# source dataset root url
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing"
HOUSING_PATH = "data/raw" # local data saving path
# specific dataset url
HOUSING_URL = DOWNLOAD_ROOT + "/housing.tgz"
# loading dataset path
data_path = "data/raw/housing.csv"

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('data/train',"train.csv")
    test_data_path: str=os.path.join('data/test',"test.csv")
    # raw_data_path: str=os.path.join('data/raw',"raw_data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def fetch_data(self, url, path):
        """
        This method will download the dataset from the source
        """
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
        # os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
        tgz_path = os.path.join(path, "raw_data.tgz")
        urllib.request.urlretrieve(url, tgz_path)
        tgz = tarfile.open(tgz_path)
        tgz.extractall(path=path)
        tgz.close()

    def load_data(self, data_path):
        """
        This method will load the dataset from the source
        """
        csv_path = os.path.join(data_path)
        return pd.read_csv(csv_path)


    def initiate_data_ingestion(self):
        """
        This method will initiate the data ingestion process 
            - save downloaded datasets in the raw folder
            - split the dataset into train and test
        """

        logging.info("Entered the data ingestion method")
        
        try:
            logging.info("Download the dataset from the source")
            self.fetch_data(HOUSING_URL, HOUSING_PATH)
            logging.info("Download completed")

            logging.info("Read raw data")
            raw_data = self.load_data(data_path)
            logging.info("Raw data saved as dataframe")

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(raw_data, test_size=0.2, random_state=42)
            logging.info("Train test split completed")
            # checking training directory exists or not
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            logging.info("Train data saved")
            # checking testing directory exists or not
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path),exist_ok=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Test data saved")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train, test = data_transformation.initiate_data_transformation(train_data,test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train,test))