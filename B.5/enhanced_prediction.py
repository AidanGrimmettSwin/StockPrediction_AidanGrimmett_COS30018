import numpy as np
import pandas as pd
import pandas_datareader as web
import datetime as dt
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler

def predict_next_k_days(model, model_inputs, scaler, prediction_prev_days, k):
    predictions = []
    real_data = model_inputs[len(model_inputs) - prediction_prev_days:, 0]
    
    for _ in range(k):
        real_data = np.reshape(real_data, (1, real_data.shape[0], 1))
        prediction = model.predict(real_data)
        prediction = scaler.inverse_transform(prediction)
        predictions.append(prediction[0][0])
        # Update the real_data with the new prediction for the next step
        real_data = np.append(real_data[0, 1:], prediction[0][0])
        
    return predictions
