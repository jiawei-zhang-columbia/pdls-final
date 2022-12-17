import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD, Adam, Nadam, Adagrad
import pickle
import time
from utils import timecallback


def prepare_data():
    """
    Prepare training, validation, and testing dataset
    """
    X_train = np.load('data/X_train_GAF_64.npy')
    y_train = np.load('data/y_train.npy')
    X_train = X_train.reshape(len(X_train), 64, 64, 1)
    mean = np.mean(X_train, axis=(1, 2), keepdims=True)
    std = np.std(X_train, axis=(1, 2), keepdims=True)
    X_train = (X_train - mean) / std
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    enc = LabelEncoder()
    enc.fit(y_train)
    y_train = enc.transform(y_train)
    y_train = to_categorical(y_train)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.8, random_state=0)

    X_test = np.load('data/X_test_GAF_64.npy')
    y_test = np.load('data/y_test.npy')
    X_test = X_test[:-6]
    y_test = y_test[:-6]
    X_test = X_test.reshape(len(X_test), 64, 64, 1)

    X_test = (X_test - mean) / std
    y_test = enc.transform(y_test)
    y_test = to_categorical(y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


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


if __name__ == '__main__':
    model_name = 'lenet'

    epochs = 50
    init_scheme = 'GlorotUniform'
    batch_size = 256
    learning_rate = 0.01389380797152978
    opt = Adagrad(learning_rate=learning_rate)

    model = build_lenet(init_scheme=init_scheme)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data()

    gen_train = ImageDataGenerator()
    gen_val = ImageDataGenerator()
    gen_test = ImageDataGenerator()

    timetaken = timecallback()
    start = time.time()
    result = model.fit(
        gen_train.flow(X_train, y_train, batch_size=batch_size),
        epochs=epochs,
        verbose=2,
        validation_data=gen_val.flow(X_val, y_val, batch_size=2048),
        callbacks=[timetaken]
    )
    end = time.time()
    with open(f'benchmark/{model_name}_history.pkl', 'wb') as handle:
        pickle.dump(result.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    model.save(f'benchmark/{model_name}')
    time_elapsed = end - start
    with open(f'benchmark/{model_name}_train_time.pkl', 'wb') as handle:
        pickle.dump(time_elapsed, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Training time: {time_elapsed} seconds')

    start = time.time()
    eval_result = model.evaluate(gen_test.flow(X_test, y_test, batch_size=2048))
    end = time.time()
    time_elapsed = end - start
    with open(f'benchmark/{model_name}_eval.pkl', 'wb') as handle:
        pickle.dump(eval_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'benchmark/{model_name}_test_time.pkl', 'wb') as handle:
        pickle.dump(time_elapsed, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Test time: {time_elapsed} seconds')
