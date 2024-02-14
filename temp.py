from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from sklearn.datasets import load_iris

# Load trained model
model = joblib.load('iris_model.pkl')
features = [[4.8,3,1.4,0.3],[2,1,3.2,1.1]]
prediction = model.predict(features)
print(prediction)
