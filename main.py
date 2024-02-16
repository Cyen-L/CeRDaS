from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import joblib
from typing import List, Union
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load trained model
model = joblib.load('iris_model.pkl')

app = FastAPI()

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class IrisFeaturesList(BaseModel):
    iris_features: List[IrisFeatures]

class BaseTrainRequest(BaseModel):
    saved_name: str = None
    random_state: int = None

class RandomForestTrainRequest(BaseTrainRequest):
    max_depth: int = None
    n_estimators: int = 100

@app.on_event("startup")
async def startup_event():
    global X_train, X_test, y_train, y_test
    ds = load_iris()
    X, y = ds.data, ds.target
    y = [ds.target_names[val] for val in y]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    

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
    predictions = model.predict(features)
    return {"predicted_species": predictions.tolist()}

@app.post("/train/random_forest/")
def train_random_forest(request: RandomForestTrainRequest):
    saved_path = os.path.join('models', request.saved_name+'.pkl')
    if os.path.exists(saved_path):
        return {"message": "Model saved name exist"}
    else:
        # Train a Random Forest Classifier1
        model = RandomForestClassifier(random_state=request.random_state, max_depth=request.max_depth, n_estimators=request.n_estimators)
        model.fit(X_train, y_train)
        joblib.dump(model, saved_path)
        return {"message": "Model trained successfully"}