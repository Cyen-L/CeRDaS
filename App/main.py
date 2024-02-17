# Import library
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import joblib
from typing import List, Union
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from io import BytesIO
from datetime import datetime
import json
from Function import *
from Classes import * # Import Classes

# Initiate FastAPI
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    global X_train, X_test, y_train, y_test
    ds = load_iris()
    X, y = ds.data, ds.target
    y = [ds.target_names[val] for val in y]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print('App Successfully Start-Up...')

@app.post("/predict/")
async def predict_species(iris_data: IrisFeaturesList):
    
    # Select the information from the Model table
    result = SQL_Query("SELECT Id, Saved_Name, Model_Version, Model_File FROM Model WHERE Saved_Name = %s AND Model_Version = %s", (iris_data.model_name, iris_data.model_version, ))
    
    # Handle user input constraint
    if len(result) == 0:
        return {"Message": "Model not exist, this can be due to incorrect model name and model version."}
    
    elif len(result) == 1:
        # Initialize variable using the SELECT result
        model_id = int(result['id'].iloc[0])
        generated_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Get the model file from DB and load into model
        model = joblib.load(BytesIO(result['model_file'].iloc[0]))
        
        # Read the features value from JSON to variable
        features = [[iris_val.sepal_length, iris_val.sepal_width, iris_val.petal_length, iris_val.petal_width] for iris_val in iris_data.iris_features]
        
        # Conduct the prediction on input features
        output_data = list(model.predict(features))

        # Convert the received JSON to compatible format
        feature_data = [json.loads(element.model_dump_json()) for element in iris_data.iris_features]
        
        # Insert the record into Model table
        SQL_Query(
            "INSERT INTO Prediction (Model_Id, Generated_Time, Feature_Data, Output_Data) VALUES (%s, %s, %s, %s)",
            (model_id, generated_time, json.dumps(feature_data), json.dumps(output_data))
            )
        
        return {"Message": "Model Prediction Successful", "Details": {"Model_Id": model_id, "Generated_Time": generated_time, "Feature_Data": feature_data, "Output_Data": output_data}}

@app.post("/train/random_forest/")
def train_random_forest(request: RandomForestTrainRequest):

    # Initialize training info
    model_algo = "Random_Forest_Classifier"
    model_parameters = request.model_parameters.model_dump_json()
    saved_name = request.saved_name
    generated_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Get model version based on number of model saved_name
    result = SQL_Query("SELECT Saved_Name, Model_Version FROM Model WHERE Saved_Name = %s", (saved_name, ))
    model_version = len(result) + 1

    # Start model training
    model = RandomForestClassifier(
        n_estimators = request.model_parameters.n_estimators,
        criterion = request.model_parameters.criterion,
        max_depth = request.model_parameters.max_depth,
        min_samples_split = request.model_parameters.min_samples_split,
        min_samples_leaf = request.model_parameters.min_samples_leaf,
        min_weight_fraction_leaf = request.model_parameters.min_weight_fraction_leaf,
        max_features = request.model_parameters.max_features,
        max_leaf_nodes = request.model_parameters.max_leaf_nodes,
        min_impurity_decrease = request.model_parameters.min_impurity_decrease,
        bootstrap = request.model_parameters.bootstrap,
        random_state = request.model_parameters.random_state) # Initialize the model
    model.fit(X_train, y_train) # Fir training data into the model

    # Convert the model into DB compatible variable
    model_file = DB_Compatible_Conversion(model)

    # Calculate the performance metrix
    balance_accuracy = balanced_accuracy_score(y_test, model.predict(X_test))

    # Insert the record into Model table
    SQL_Query(
        "INSERT INTO Model (Model_Algo, Model_Parameters, Saved_Name, Model_Version, Balance_Accuracy, Generated_Time, Model_File) VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (model_algo, model_parameters, saved_name, model_version, balance_accuracy, generated_time, model_file)
        )
    
    return {"Message": "Model Training Successful", "Details": {"Model": model_algo, "Model_Version": model_version, "Model_Parameters": json.loads(model_parameters), "Balance_Accuracy": balance_accuracy, "Generated_Time": generated_time}}

@app.post("/train/gaussian_naive_bayes/")
def train_gaussian_naive_bayes(request: GaussianNaiveBayesTrainRequest):
    
    # Initialize training info
    model_algo = "Gaussian Naive Bayes"
    model_parameters = request.model_parameters.model_dump_json()
    saved_name = request.saved_name
    generated_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Get model version based on number of model saved_name
    result = SQL_Query("SELECT Saved_Name, Model_Version FROM Model WHERE Saved_Name = %s", (saved_name, ))
    model_version = len(result) + 1

    # Start model training
    model = GaussianNB() # Initialize the model
    model.fit(X_train, y_train) # Fir training data into the model

    # Convert the model into DB compatible variable
    model_file = DB_Compatible_Conversion(model)

    # Calculate the performance metrix
    balance_accuracy = balanced_accuracy_score(y_test, model.predict(X_test))

    # Insert the record into Model table
    SQL_Query(
        "INSERT INTO Model (Model_Algo, Model_Parameters, Saved_Name, Model_Version, Balance_Accuracy, Generated_Time, Model_File) VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (model_algo, model_parameters, saved_name, model_version, balance_accuracy, generated_time, model_file)
        )
    
    return {"Message": "Model Training Successful", "Details": {"Model": model_algo, "Model_Version": model_version, "Model_Parameters": json.loads(model_parameters), "Balance_Accuracy": balance_accuracy, "Generated_Time": generated_time}}


@app.post("/train/logistic_regression/")
def train_logistic_regression(request: LogisticRegressionTrainRequest):
    
    # Initialize training info
    model_algo = "Logistic_Regression"
    model_parameters = request.model_parameters.model_dump_json()
    saved_name = request.saved_name
    generated_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Get model version based on number of model saved_name
    result = SQL_Query("SELECT Saved_Name, Model_Version FROM Model WHERE Saved_Name = %s", (saved_name, ))
    model_version = len(result) + 1

    # Start model training
    model = LogisticRegression(
        penalty=request.model_parameters.penalty, 
        dual=request.model_parameters.dual, 
        tol=request.model_parameters.tol, 
        C=request.model_parameters.C, 
        fit_intercept=request.model_parameters.fit_intercept, 
        intercept_scaling=request.model_parameters.intercept_scaling, 
        random_state=request.model_parameters.random_state, 
        solver=request.model_parameters.solver) # Initialize the model
    model.fit(X_train, y_train) # Fir training data into the model

    # Convert the model into DB compatible variable
    model_file = DB_Compatible_Conversion(model)

    # Calculate the performance metrix
    balance_accuracy = balanced_accuracy_score(y_test, model.predict(X_test))

    # Insert the record into Model table
    SQL_Query(
        "INSERT INTO Model (Model_Algo, Model_Parameters, Saved_Name, Model_Version, Balance_Accuracy, Generated_Time, Model_File) VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (model_algo, model_parameters, saved_name, model_version, balance_accuracy, generated_time, model_file)
        )
    
    return {"Message": "Model Training Successful", "Details": {"Model": model_algo, "Model_Version": model_version, "Model_Parameters": json.loads(model_parameters), "Balance_Accuracy": balance_accuracy, "Generated_Time": generated_time}}
