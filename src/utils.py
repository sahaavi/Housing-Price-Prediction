import os
import sys

import numpy as np
import pickle
# from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import cross_val_score

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test, y_test, models, params):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            # using model params
            param=params[list(models.keys())[i]]

            # model fitting without params
            # model.fit(X_train,y_train)

            # model fitting with params using grid search cross validation
            grid_search = GridSearchCV(model, param, cv=2)
            grid_search.fit(X_train, y_train)

            # model fitting with best params
            model.set_params(**grid_search.best_params_)
            model.fit(X_train, y_train)

            # y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            # train_model_score = r2_score(y_train, y_train_pred)

            # r_square
            # test_model_rsquare = r2_score(y_test, y_test_pred)

            # rmse
            test_model_mse = mean_squared_error(y_test, y_test_pred)
            test_model_rmse = np.sqrt(test_model_mse)

            report[list(models.keys())[i]] = test_model_rmse

        return report
    
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)

        return obj
    
    except Exception as e:
        raise CustomException(e, sys)