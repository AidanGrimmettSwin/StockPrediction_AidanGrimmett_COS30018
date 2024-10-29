import numpy as np
import pandas as pd
import pandas_datareader as web
import datetime as dt
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler

def predict_next_k_days(model, model_inputs, scaler, prediction_prev_days, k):
    real_data = [model_inputs[-prediction_prev_days:]]  # Extract the last prediction_prev_days of model inputs

    real_data = np.array(real_data)  # Convert to np array
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], real_data.shape[2]))  # Ensure 3D shape with extra features
    
    predictions = []

    for _ in range(k):
        prediction = model.predict(real_data)  # Predict the next day
        predictions.append(prediction) # Add new prediction
        # Update the real_data with the new prediction (to predict the next day after that)
        new_data = np.append(real_data[0][1:], prediction, axis=0)  # Remove the first timestep and append the new prediction
        real_data = np.array([new_data])
        
    predictions = np.array(predictions)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))

    return predictions