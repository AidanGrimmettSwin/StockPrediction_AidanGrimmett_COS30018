from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN, GRU

def MakeCustomModel(model_type, input_data, n_layers, units, dropout):
    
    #dictionary to map model_type to the appropriate Keras class
    model_layer_dict = {
        'LSTM': LSTM,
        'RNN': SimpleRNN,
        'GRU': GRU
    }

    #if not valid model type is parsed, select LSTM as default
    if (model_type not in model_layer_dict):
        model_type = LSTM

    
    #create a blank sequential model that we will add layers to
    model = Sequential()

    for i in range(n_layers - 1):
        if (i == 0):
            #Add first layer, specify input shape which is not needed in future layers
            model.add(model_layer_dict[model_type](units, return_sequences=True, input_shape = input_data))
        else:
            #add the rest of the layers
            model.add(model_layer_dict[model_type](units, return_sequences=True))
        
        # add dropout on each layer, prevents overfitting by randomly dropping some neurons
        model.add(Dropout(dropout))


    # add final layer, dont need to return sequences as this is the output layer
    model.add(model_layer_dict[model_type](units, return_sequences=False))

    #specify loss function, mean absolute error punishes severe miscalculations
    loss = "mean_absolute_error"
    #optimizer minimizes the loss and updates the weights
    optimizer="rmsprop"
    # add output layer
    model.add(Dense(1, activation="linear"))
    #compile the model and specify the loss, optimizer and metrics to track
    model.compile(loss=loss, metrics = [loss], optimizer = optimizer)
    return model #return complete model

