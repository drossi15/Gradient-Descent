


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gradient import gradient_descent
from model import build_model_data,standardize




df = pd.read_csv("dataset/NY-House-Dataset.csv")
columns_to_drop = ["ADDRESS", "STATE", "MAIN_ADDRESS", "ADMINISTRATIVE_AREA_LEVEL_2", 
                   "LOCALITY", "STREET_NAME", "LONG_NAME", "FORMATTED_ADDRESS", 
                   "LATITUDE", "LONGITUDE","TYPE","SUBLOCALITY","BROKERTITLE"]
df_filtered = df.drop(columns=columns_to_drop)
#FILTRO GLI OUTLIER
vars = ["PRICE", "BEDS", "BATH", "PROPERTYSQFT"]
# Filtra i valori anomali utilizzando il concetto di IQR
for var in vars:
    Q1 = df_filtered[var].quantile(0.25)
    Q3 = df_filtered[var].quantile(0.75)
    
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    
    df_filtered = df_filtered[(df_filtered[var] >= lower_limit) & (df_filtered[var] <= upper_limit)]
    
###########################################################################################################ààà

features = df_filtered[['BEDS','BATH']]
prices = df_filtered['PRICE']
print(features)
print(prices)

A, b = build_model_data(standardize(features), standardize(prices))


cond_number = np.linalg.cond(A)
print("Numero di condizionamento di A:", cond_number)

max_iters = 500
gamma = 0.00001  # Tasso di apprendimento iniziale
x_initial = np.zeros(A.shape[1])



# Start gradient descent.
gradient_objectives_naive, gradient_xs_naive = gradient_descent(A, x_initial, b, max_iters, gamma)
print("########################")