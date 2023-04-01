# Predicting Housing Prices using Machine Learning

This project is an end-to-end machine learning project that involves building a model to predict housing prices using the California Housing dataset. The project covers all stages of a typical machine learning project, including data exploration, preparation, feature engineering, model selection and evaluation, and deployment.

## Details

**requirements.txt**
> Here `-e .` will triger the setup.py file.

**components**
> This sub-package is for data ingestion, transformation and model training.  
`data_ingestion` Data ingestion is the process of importing large, assorted data files from multiple sources into a single, cloud-based storage medium—a data warehouse, data mart or database—where it can be accessed and analyzed. Also use this module to split the dataset into train and test.  
`data_transformation` Data transformation is the process of converting data from one format or structure into another format or structure. For example, change categoriacal features into numerical features, one hot encoding, label encoding.  
`model_trainer` Here we'll train our models

**pipeline**
> This sub-package is for pipeline. We'll use two pipelines.  
> - `train_pipeline` This module is for training pipeline. This will triger the modules of components. 
> - `predict_pipeline` This module is for prediction purpose. 

**logger.py**

> Deals with logging. In commercial software products logging is actually crucial because logging allows to detect bugs sooner, it allows to traceback easily when a problem occurs.

**exception.py**

> Deals with exception handling.  
`exc.info()` This function returns the old-style representation of the handled exception. If an exception e is currently handled (so exception() would return e), exc_info() returns the tuple (type(e), e, e.\__traceback__). That is, a tuple containing the type of the exception (a subclass of BaseException), the exception itself, and a traceback object which typically encapsulates the call stack at the point where the exception last occurred.  
If no exception is being handled anywhere on the stack, this function return a tuple containing three None values.

**utils.py**

> Store common funtions to use anywhere in the app. We'll basically try to use these function inside the components modules.

## Acknowledgments

Some steps of this project built by following the steps provided in the book "[Hands-On Machine Learning with Scikit-Learn and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781491962282/)" by [Aurélien Géron](https://github.com/ageron). The book provided clear and detailed explanations of each step, along with practical tips and best practices, and it used popular libraries such as Pandas, NumPy, Scikit-Learn, and TensorFlow to implement the code. 

Book GitHub repo link: https://github.com/ageron/handson-ml3