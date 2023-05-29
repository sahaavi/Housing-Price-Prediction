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

            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "AdaBoost Classifier":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
            }

            model_report:dict=evaluate_models(X_train=X_train, 
                                              y_train=y_train,
                                              X_test=X_test,
                                              y_test=y_test,
                                              models=models,
                                              params=params)

            # get the besq model score
            # best_model_score = max(sorted(model_report.values())) # for r2_score
            best_model_score = min(sorted(model_report.values())) # for root mean_squared_error
            # get the best model name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                ]
            logging.info(f"Best model name: {best_model_name}")
            best_model = models[best_model_name]

            # for r2_score
            # if best_model_score < 0.6:
            #     raise CustomException("Best model score is less than 0.6")
            
            logging.info("Indexed the best model")

            save_object(
                file_path = self.model_trainer_config.model_obj_path,
                obj = best_model
            )

            # prediction = best_model.predict(X_test)
            # r_square = r2_score(y_test, prediction)
            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)