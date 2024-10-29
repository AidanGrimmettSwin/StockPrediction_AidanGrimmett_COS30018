import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def ensemble_modeling(data, test_data, scalers, second_model, prediction_days, weights=(0.5, 0.5)):
    # train the ARIMA model

    # fit the ARIMA model on the training data
    arima_model = ARIMA(scalers['Close'].inverse_transform(data['Close'].values.reshape(-1, 1)), order=(5, 1, 0))
    arima_result = arima_model.fit()


    # predict on the test set using ARIMA
    arima_forecast = arima_result.forecast(steps=len(test_data))

    # predict with original model

    x_test = []
    for x in range(prediction_days, len(test_data)):
        x_test.append(test_data.iloc[x - prediction_days:x].values)

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))
    second_model_predictions = second_model.predict(x_test)
    second_model_predictions = scalers['Close'].inverse_transform(second_model_predictions)

    # combine predictions

    # ARIMA forecast and second_model predictions need to be the same length for ensemble
    arima_forecast = np.array(arima_forecast[:len(second_model_predictions)]).reshape(-1, 1)

    # calculate ensemble predictions using specified weights
    ensemble_predictions = (weights[0] * arima_forecast) + (weights[1] * second_model_predictions)
    return ensemble_predictions
