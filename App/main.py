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

# Load trained model
#model = joblib.load('iris_model.pkl')

app = FastAPI()

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class IrisFeaturesList(BaseModel):
    model_name: str = None
    iris_features: List[IrisFeatures]

class BaseTrainRequest(BaseModel):
    saved_name: str = None

class RandomForestTrainRequest(BaseTrainRequest):
    max_depth: int = None
    n_estimators: int = 100
    random_state: int = None

class LogisticRegressionTrainRequest(BaseTrainRequest):
    penalty: str = 'l2'
    dual: bool = False
    tol: float = 1e-4
    C: float = 1.0
    fit_intercept: bool = True
    intercept_scaling: float = 1.0
    random_state: int = None
    solver: str = 'lbfgs'


@app.on_event("startup")
async def startup_event():
    global X_train, X_test, y_train, y_test
    ds = load_iris()
    X, y = ds.data, ds.target
    y = [ds.target_names[val] for val in y]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print('App Successfully Start-Up...')
    

@app.post("/predict_single/")
async def predict_species(iris_data: IrisFeatures):
    features = [[iris_data.sepal_length, iris_data.sepal_width, iris_data.petal_length, iris_data.petal_width]]
    prediction = model.predict(features)
    return {"predicted_species": prediction[0]}

@app.post("/predict_multi/")
async def predict_species(iris_data: IrisFeaturesList):
    features = [[iris_val.sepal_length, iris_val.sepal_width, iris_val.petal_length, iris_val.petal_width] for iris_val in iris_data.iris_features]
    predictions = model.predict(features)
    return {"predicted_species": predictions.tolist()}

@app.post("/predict/")
async def predict_species(iris_data: Union[IrisFeatures, IrisFeaturesList]):
    if isinstance(iris_data, IrisFeatures):
        features = [[iris_data.sepal_length, iris_data.sepal_width, iris_data.petal_length, iris_data.petal_width]]
    elif isinstance(iris_data, IrisFeaturesList):
        features = [[iris_val.sepal_length, iris_val.sepal_width, iris_val.petal_length, iris_val.petal_width] for iris_val in iris_data.iris_features]
    else:
        return {'Fromat Error'}
    if iris_data.model_name:
        model_path = os.path.join('models', iris_data.model_name + '.pkl')
        if not os.path.exists(model_path):
            return {"Message": "Model not exist", "Model_Path": str(model_path)}
    else:
        model_path = 'iris_model.pkl'
    joblib.load(model_path)
    predictions = model.predict(features)
    return {"Message": "Model Prediction Successful", "Model_Path":model_path, "Predicted_Species": predictions.tolist()}

@app.post("/train/random_forest/")
def train_random_forest(request: RandomForestTrainRequest):
    saved_path = os.path.join('models', request.saved_name + '.pkl')
    if os.path.exists(saved_path):
        return {"Message": "Model saved name exist"}
    else:
        # Train a Random Forest Classifier1
        model = RandomForestClassifier(random_state=request.random_state, max_depth=request.max_depth, n_estimators=request.n_estimators)
        model.fit(X_train, y_train)
        joblib.dump(model, saved_path)
        return {"Message": "Model Training Successful", "Model":"Random Forest Classifier", "Saved_Path": saved_path}

@app.post("/train/gaussian_naive_bayes/")
def train_gaussian_naive_bayes(request: BaseTrainRequest):
    saved_path = os.path.join('models', request.saved_name + '.pkl')
    if os.path.exists(saved_path):
        return {"Message": "Model saved name exist"}
    else:
        # Train a Random Forest Classifier1
        model = GaussianNB()
        model.fit(X_train, y_train)
        joblib.dump(model, saved_path)
        return {"Message": "Model Training Successful", "Model": "Gaussian Naive Bayes", "Saved_Path": saved_path}

@app.post("/train/logistic_regression/")
def train_logistic_regression(request: LogisticRegressionTrainRequest):
    saved_path = os.path.join('models', request.saved_name + '.pkl')
    if os.path.exists(saved_path):
        return {"Message": "Model saved name exist"}
    else:
        # Train a Random Forest Classifier1
        model = LogisticRegression(penalty=request.penalty, dual=request.dual, tol=request.tol, C=request.C, fit_intercept=request.fit_intercept, intercept_scaling=request.intercept_scaling, random_state=request.random_state, solver=request.solver)
        model.fit(X_train, y_train)
        joblib.dump(model, saved_path)
        return {"Message": "Model Training Successful", "Model": "Logistic Regression", "Saved_Path": saved_path}