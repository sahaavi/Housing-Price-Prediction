import os
import sys

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.impute import SimpleImputer 
from sklearn.base import BaseEstimator, TransformerMixin

from src.exception import CustomException 
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    # storing pickle file path
    preprocessor_obj_path: str=os.path.join('data/model',"preprocessor.pkl")

class DataFrameSelector(BaseEstimator, TransformerMixin):
    """
    This class will select the required columns from the dataset
    """
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
    
class MyLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)
    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self
    def transform(self, x, y=0):
        return self.encoder.transform(x)

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This method will return the data transformer object
        """
        try:
            numerical_columns = ['longitude', 'latitude', 'housing_median_age', 
                                 'total_rooms', 'total_bedrooms', 'population', 
                                 'households', 'median_income']
            categorical_columns = ['ocean_proximity']

            num_pipeline = Pipeline([
                ('selector', DataFrameSelector(numerical_columns)),
                ('imputer', SimpleImputer(strategy="median")),
                ('std_scaler', StandardScaler()),
            ])
            logging.info("Numerical pipeline created")
            logging.info(f"Numerical columns: {numerical_columns}")

            cat_pipeline = Pipeline([
                ('selector', DataFrameSelector(categorical_columns)),
                ('label_binarizer', MyLabelBinarizer()),
            ])
            logging.info("Categorical pipeline created")
            logging.info(f"Categorical columns: {categorical_columns}")

            preprocessor = FeatureUnion(transformer_list=[
                ("num_pipeline", num_pipeline),
                ("cat_pipeline", cat_pipeline),
            ])
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and test data loaded successfully")

            target_feature_train = train_df["median_house_value"].copy()
            target_feature_test = test_df["median_house_value"].copy()

            input_feature_train_df=train_df.drop(columns=["median_house_value"],axis=1)
            input_feature_test_df=test_df.drop(columns=["median_house_value"],axis=1)
        
            preprocessor = self.get_data_transformer_object()
            logging.info("Initiated data transformation")

            train_prepared = preprocessor.fit_transform(input_feature_train_df)
            logging.info("Train data transformed successfully")

            test_prepared = preprocessor.fit_transform(input_feature_test_df)
            logging.info("Test data transformed successfully")

            save_object(
                # function to save the object - src/utils/common.py
                file_path = self.data_transformation_config.preprocessor_obj_path,
                obj = preprocessor
            )

            logging.info("Preprocessor object saved successfully as Pcikle file")

            train_arr = np.c_[train_prepared, np.array(target_feature_train)]
            test_arr = np.c_[test_prepared, np.array(target_feature_test)]

            return train_arr, test_arr

        except Exception as e:
            raise CustomException(e, sys)