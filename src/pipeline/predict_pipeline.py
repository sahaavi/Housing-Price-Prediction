import sys

import pandas as pd

from src.exception import CustomException
from src.utils import load_object
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'data/model/model.pkl'
            preprocessor_path = 'data/model/preprocessor.pkl'
            # load model and preprocessor
            logging.info('Loading model and preprocessor in prediction pipeline')
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            logging.info('Loaded model and preprocessor')
            # scale features
            features_scaled = preprocessor.transform(features)
            logging.info('Scaled features')
            # predict
            logging.info('Predicting')
            prediction = model.predict(features_scaled)
            logging.info('Predicted')
            return prediction
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 longitude: float,
                 latitude: float,
                 housing_median_age: float,
                 total_rooms: float,
                 total_bedrooms: float,
                 population: float,
                 households: float,
                 median_income: float,
                 ocean_proximity: str):
        
        self.longitude = longitude
        self.latitude = latitude
        self.housing_median_age = housing_median_age
        self.total_rooms = total_rooms
        self.total_bedrooms = total_bedrooms
        self.population = population
        self.households = households
        self.median_income = median_income
        self.ocean_proximity = ocean_proximity
    
    def to_df(self):
        try:
            input_data_dict = {
                'longitude': [self.longitude],
                'latitude': [self.latitude],
                'housing_median_age': [self.housing_median_age],
                'total_rooms': [self.total_rooms],
                'total_bedrooms': [self.total_bedrooms],
                'population': [self.population],
                'households': [self.households],
                'median_income': [self.median_income],
                'ocean_proximity': [self.ocean_proximity]
            }
            return pd.DataFrame(input_data_dict)
        except Exception as e:
            raise CustomException(e, sys)