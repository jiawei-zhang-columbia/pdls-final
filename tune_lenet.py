import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.utils import shuffle
from tensorflow.keras.optimizers import SGD, Adam, Nadam, Adagrad
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from hyperas.distributions import choice, uniform
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from sklearn.model_selection import train_test_split
import hyperopt as hp
from hyperas.distributions import uniform, loguniform
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
import gc
from keras import backend as K


def data():
    """
    Data function to split and preprocess data
    Returns training and validation dataset
    """
    X_train = np.load('data/X_train_GAF_64.npy')
    y_train = np.load('data/y_train.npy')
    X_train = X_train.reshape(len(X_train), 64, 64, 1)
    mean = np.mean(X_train, axis=(1,2), keepdims=True)
    std = np.std(X_train, axis=(1,2), keepdims=True)
    X_train = (X_train - mean) / std
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    enc = LabelEncoder()
    enc.fit(y_train)
    y_train = enc.transform(y_train)
    y_train = to_categorical(y_train)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.8, random_state=0)

    return X_train, y_train, X_val, y_val


def create_model(X_train, y_train, X_val, y_val):
    def build_lenet(init_scheme):
        model = Sequential()
        model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 1), kernel_initializer=init_scheme))
        model.add(AveragePooling2D())
        model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', kernel_initializer=init_scheme))
        model.add(AveragePooling2D())
        model.add(Flatten())
        model.add(Dense(units=120, activation='relu', kernel_initializer=init_scheme))
        model.add(Dense(units=84, activation='relu', kernel_initializer=init_scheme))
        model.add(Dense(units=6, activation='softmax'))
        return model

    init_scheme = {{choice(['HeNormal', 'GlorotUniform'])}}
    model = build_lenet(init_scheme)

    batch_size = {{choice([256, 512, 1024, 2048])}}
    optimizer = {{choice(['Adam', 'SGD', 'Nadam', 'Adagrad'])}}

    learning_rate = {{loguniform(np.log(1e-5), np.log(1.))}}
    if optimizer == 'Adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'SGD':
        opt = SGD(learning_rate=learning_rate, momentum=0.8, decay=1e-4)
    elif optimizer == 'Nadam':
        opt = Nadam(learning_rate=learning_rate)
    else:
        opt = Adagrad(learning_rate=learning_rate)

    gen_train = ImageDataGenerator()
    gen_val = ImageDataGenerator()
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
              optimizer=opt)
    result = model.fit(gen_train.flow(X_train, y_train, batch_size=batch_size),
              epochs=50,
              verbose=2,
              validation_data=gen_val.flow(X_val, y_val, batch_size=2048)
            )
    validation_acc = np.amax(result.history['val_accuracy'])
    # print(learning_rate)
    K.clear_session()
    gc.collect()
    print('Best validation acc of trial:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'val_accuracy': validation_acc,
           'init_scheme': init_scheme, 'optimizer': optimizer, 'learning_rate': learning_rate,
           'batch_size': batch_size}


if __name__ == '__main__':
    start = time.time()
    max_evals = 50
    model_name = 'lenet'
    trials = Trials()
    best_run, best_model = optim.minimize(model=create_model,
                                      data=data,
                                      algo=tpe.suggest,
                                      max_evals=max_evals,
                                      trials=trials)
    print(trials)
    with open(f'history/{model_name}.pkl', 'wb') as  handle:
        pickle.dump(trials, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'runs/{model_name}.pkl', 'wb') as handle:
        pickle.dump(best_run, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(best_run)
    end = time.time()
    print(f"Time elpased: {end-start} seconds")