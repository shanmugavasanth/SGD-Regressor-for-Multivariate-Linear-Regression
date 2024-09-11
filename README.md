# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
### Step 1: Start 
### Step 2: Data Preparation 
### Step 3: Hypothesis Definition 
### Step 4: Cost Function 
### Step 5: Parameter Update Rule 
### Step 6: Iterative Training 
### Step 7: Model Evaluation 
### Step 8: End

## Program:

Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.

Developed by: Shanmuga Vasanth M

RegisterNumber: 212223040191

```
import numpy as np 
import pandas as pd 
from sklearn.datasets import fetch_california_housing 
from sklearn. linear_model import SGDRegressor 
from sklearn.multioutput import MultiOutputRegressor 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error 
from sklearn.preprocessing import StandardScaler
dataset = fetch_california_housing()
df=pd. DataFrame(dataset.data, columns=dataset. feature_names)
df ['HousingPrice']=dataset.target
print(df.head ())
X = df. drop(columns=['AveOccup', 'HousingPrice'])#data[:, :3]
Y = df[[ 'AveOccup', 'HousingPrice' ]]
```

## Output:

![Screenshot 2024-09-11 085827](https://github.com/user-attachments/assets/2891e293-b10d-4631-897e-3fcd14f30e61)

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
