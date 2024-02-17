# Import library
from pydantic import BaseModel
from typing import List, Union

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class IrisFeaturesList(BaseModel):
    model_name: str
    model_version: int
    iris_features: List[IrisFeatures]

class BaseTrainRequest(BaseModel):
    saved_name: str = None

class RandomForestParameters(BaseModel):
    n_estimators: int = 100
    criterion: str = "gini"
    max_depth: int = None
    min_samples_split: Union[int, float] = 2
    min_samples_leaf: Union[int, float] = 1
    min_weight_fraction_leaf: float = 0.0
    max_features: Union[int, float, str] = 'sqrt'
    max_leaf_nodes: int = None
    min_impurity_decrease: float = 0.0
    bootstrap: bool = True
    random_state: int = None

class RandomForestTrainRequest(BaseTrainRequest):
    model_parameters: RandomForestParameters

class GaussianNaiveBayesParameters(BaseModel):
    var_smoothing: float = 1e-9

class GaussianNaiveBayesTrainRequest(BaseTrainRequest):
    model_parameters: GaussianNaiveBayesParameters

class LogisticRegressionParameters(BaseTrainRequest):
    penalty: str = 'l2'
    dual: bool = False
    tol: float = 1e-4
    C: float = 1.0
    fit_intercept: bool = True
    intercept_scaling: float = 1.0
    random_state: int = None
    solver: str = 'lbfgs'

class LogisticRegressionTrainRequest(BaseTrainRequest):
    model_parameters: LogisticRegressionParameters