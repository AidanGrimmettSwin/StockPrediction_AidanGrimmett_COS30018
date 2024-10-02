# File: stock_prediction.py
# Authors: Bao Vo and Cheong Koo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 02/07/2024 (v3)

# Code modified from:
# Title: Predicting Stock Prices with Python
# Youtuble link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine

# Need to install the following (best in a virtual env):
# pip install numpy
# pip install matplotlib
# pip install pandas
# pip install tensorflow
# pip install scikit-learn
# pip install pandas-datareader
# pip install yfinance

# Can't run without specifying output encoding
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer
import yfinance as yf
import os

from load_process_data import load_and_process_data
from plot_functions import plot_boxplot, plot_candlestick
from CreateCustomModel import MakeCustomModel
from predict_k_days import predict_next_k_days

if not os.path.isdir("data"):
    os.mkdir("data")


# DATA_SOURCE = "yahoo"
COMPANY = 'CBA.AX'

#load data if it has been retrieved before, else download it and store it
data, test_data, scalers  = load_and_process_data(COMPANY, True, True, True, '/Data')


PRICE_VALUE = "Close"



# Number of days to look back to base the prediction
PREDICTION_DAYS = 60 # Original

# To store the training data
x_train = []
y_train = []

#scaled_data = scaled_data[:,0] # Turn the 2D array back to a 1D array

# Prepare the data
for x in range(PREDICTION_DAYS, len(data)):
    x_train.append(data.iloc[x-PREDICTION_DAYS:x].values)
    y_train.append(data.iloc[x]['Close'])

# Convert them into an array
x_train, y_train = np.array(x_train), np.array(y_train)
# Now, x_train is a 2D array(p,q) where p = len(scaled_data) - PREDICTION_DAYS
# and q = PREDICTION_DAYS; while y_train is a 1D array(p)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
# We now reshape x_train into a 3D array(p, q, 1); Note that x_train 
# is an array of p inputs with each input being a 2D array 


#------------------------------------------------------------------------------
# Build the Model

input_shape=(x_train.shape[1], x_train.shape[2])

model = MakeCustomModel('LSTM', input_shape, 2, 256, 0.2)

model.fit(x_train, y_train, epochs=20, batch_size=64)

# Scale the actual test prices correctly for plotting
actual_prices = test_data[PRICE_VALUE].values.reshape(-1, 1)
actual_prices = scalers['Close'].inverse_transform(actual_prices)

total_dataset = pd.concat((data[PRICE_VALUE], test_data[PRICE_VALUE]), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - PREDICTION_DAYS:].values
# We need to do the above because to predict the closing price of the fisrt
# PREDICTION_DAYS of the test period [TEST_START, TEST_END], we'll need the 
# data from the training period

model_inputs = model_inputs.reshape(-1, 1)
# TO DO: Explain the above line

model_inputs = scalers['Close'].transform(model_inputs)
# We again normalize our closing price data to fit them into the range (0,1)
# using the same scaler used above 
# However, there may be a problem: scaler was computed on the basis of
# the Max/Min of the stock price for the period [TRAIN_START, TRAIN_END],
# but there may be a lower/higher price during the test period 
# [TEST_START, TEST_END]. That can lead to out-of-bound values (negative and
# greater than one)
# We'll call this ISSUE #2

# TO DO: Generally, there is a better way to process the data so that we 
# can use part of it for training and the rest for testing. You need to 
# implement such a way

#------------------------------------------------------------------------------
# Make predictions on test data
#------------------------------------------------------------------------------
x_test = []
for x in range(PREDICTION_DAYS, len(test_data)):
    x_test.append(test_data.iloc[x - PREDICTION_DAYS:x].values)

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))
# TO DO: Explain the above 5 lines

predicted_prices = model.predict(x_test)
predicted_prices = scalers['Close'].inverse_transform(predicted_prices)
# Clearly, as we transform our data into the normalized range (0,1),
# we now need to reverse this transformation 
#------------------------------------------------------------------------------
# Plot the test predictions

plot = input("Input: \n1 for standard graph\n2 for box plot\n3 for candlestick plot")
if plot == '1':

    plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
    plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
    plt.title(f"{COMPANY} Share Price")
    plt.xlabel("Time")
    plt.ylabel(f"{COMPANY} Share Price")
    plt.legend()
    plt.show()
elif plot == '2':
    plot_boxplot(data, window_size=90, title="90 Day window box plot")
elif plot == '3':
    plot_candlestick(data, str(COMPANY + " candlestick plot"), 30)

#------------------------------------------------------------------------------
# Predict next day(s)
#------------------------------------------------------------------------------

if (True):
    # Number of days to predict
    k = int(input("Enter the number of days you want to predict: "))

    # Make predictions for the next k days
    predictions = predict_next_k_days(model, model_inputs, scalers, PREDICTION_DAYS, k)

    # Display the predictions
    print(f"Predicted prices for the next {k} days: {predictions}")

    # Plot predictions
    plt.plot(range(1, k + 1), predictions, color="green", label=f"Predicted {COMPANY} Price")
    plt.title(f"{COMPANY} Share Price Prediction for the Next {k} Days")
    plt.xlabel("Days")
    plt.ylabel(f"{COMPANY} Share Price")
    plt.legend()
    plt.show()
else:
    real_data = [model_inputs[len(model_inputs) - PREDICTION_DAYS:, 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    print(f"Prediction: {prediction}")