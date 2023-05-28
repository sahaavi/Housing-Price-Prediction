import os
import sys

from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import r2_score

from src.exception import CustomException 
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    # storing pickle file path
    model_obj_path: str=os.path.join('data/model',"model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train, test):
        try:
            logging.info("Initiating model training")
            X_train, y_train, X_test, y_test = (
                train[:, :-1], # all rows, no label
                train[:, -1], # all rows, label only
                test[:, :-1], # all rows, no label
                test[:, -1], # all rows, label only
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "AdaBoost Classifier": AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
            }

            model_report:dict=evaluate_models(X_train=X_train, 
                                              y_train=y_train,
                                              X_test=X_test,
                                              y_test=y_test,
                                              models=models)

            # get the besq model score
            best_model_score = max(sorted(model_report.values()))
            # get the best model name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                ]
            logging.info(f"Best model name: {best_model_name}")
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("Best model score is less than 0.6")
            
            logging.info("Indexed the best model")

            save_object(
                file_path = self.model_trainer_config.model_obj_path,
                obj = best_model
            )

            # prediction = best_model.predict(X_test)
            # r2_square = r2_score(y_test, prediction)
            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)