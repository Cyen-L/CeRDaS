from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

# Load trained model
model = joblib.load('iris_model.pkl')

app = FastAPI()

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict/")
async def predict_species(iris: IrisFeatures):
    features = [[iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]]
    prediction = model.predict(features)
    print(prediction[0])
    return {"predicted_species": int(prediction[0])}