from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from typing import List

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

@app.post("/predict_single/")
async def predict_species(iris: IrisFeatures):
    features = [[iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]]
    prediction = model.predict(features)
    print(prediction)
    return {"predicted_species": prediction[0]}


@app.post("/predict_multi/")
async def predict_species(iris_list: List[IrisFeatures]):
    features = [[iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width] for iris in iris_list]
    predictions = model.predict(features)
    return {"predicted_species": predictions.tolist()}