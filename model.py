import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo

# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 
  
# data (as pandas dataframes) 
X = heart_disease.data.features 
y = heart_disease.data.targets 
  
# metadata 
print(heart_disease.metadata) 
  
# variable information 
print(heart_disease.variables) 

# normalise age,trestbps,chol,thalach,oldpeak,ca

# randomly shuffle dataset

# split dataset 20% for testing, 20% for validation, 60% for training

# train model

# test model

# get accuracy, precision and recall