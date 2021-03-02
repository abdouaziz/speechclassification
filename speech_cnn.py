import numpy as np 
import matplotlib.pyplot as plt
import json 
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.optimizers import Adam , RMSprop

PATH_DATA= 'data.json'

def load_data(file =PATH_DATA):
    with open(file , 'r') as f:
        data= json.load(f)

        X = np.array(data["mffc"])
        y= np.array(data["labels"])

    return X,y


def prepare_datasets(test_size , validation_size):

    X,y =  load_data()

    X_train , X_test , y_train , y_test =  train_test_split(X, y , test_size=test_size)
    X_train , X_validation , y_train , y_validation = train_test_split(X_train , y_train , test_size=validation_size)


    #3D here we convert the dimension of our array

    X_train = X_train[... , np.newaxis]
    X_validation = X_validation[... , np.newaxis]
    X_test= X_test[... , np.newaxis]


    return X_train , X_validation , X_test , y_train , y_validation , y_test

#We can do a transfert learning if we need

def build_model(input_shape):

    # build network topology
    model = keras.Sequential()

    # 1st conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 2nd conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # flatten output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(10, activation='softmax'))

    model.summary()

    return model 


""" 
def build_model(input_shape):
   
    # build network topology
    model = keras.Sequential()

    # 2 LSTM layers
    model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.LSTM(64))

    # dense layer
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model    """

def predict(model, X, y):
    mapping= [
        "blues",
        "classical",
        "country",
        "disco",
        "hiphop",
        "jazz",
        "metal",
        "pop",
        "reggae",
        "rock"
    ]
   
    # add a dimension to input data for sample - model.predict() expects a 4d array in this case
    X = X[np.newaxis, ...] # array shape (1, 130, 13, 1)

    # perform prediction
    prediction = model.predict(X)

    # get index with max value
    predicted_index = np.argmax(prediction, axis=1)

    print("Target: {}, Predicted label: {}".format(y, predicted_index))
    print("The class is : {}".format(mapping[predicted_index[0]]))

def plot_graph(history , string):
  plt.plot(history.history[string])
  plt.plot(history.history["val_"+string])
  plt.xlabel(string)
  plt.ylabel("val_"+string)
  plt.legend([string , "val_"+string])
  plt.show()





if __name__ == "__main__":
    X_train , X_validation , X_test , y_train , y_validation , y_test = prepare_datasets(0.25 , 0.2) 

    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = build_model(input_shape) 
    model.compile(loss ="sparse_categorical_crossentropy" , optimizer=Adam(learning_rate=0.001) , metrics=["accuracy"])

    history=model.fit(X_train ,y_train , batch_size=32 , epochs=100 , validation_data=(X_validation , y_validation)) 

    plot_graph(history , "accuracy")
    plot_graph(history , "loss")
    #Here we evaluate our mode

    test_error , test_accuracy = model.evaluate(X_test , y_test , verbose=1)

    print("{} Accuracy on the test set".format(test_accuracy)) 

    X_to_predict = X_test[100]
    y_to_predict = y_test[100]

    # predict sample
    predict(model, X_to_predict, y_to_predict)
    



 