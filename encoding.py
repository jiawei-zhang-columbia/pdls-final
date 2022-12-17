import numpy as np
import arff
from pyts.image import GramianAngularField


dataset = arff.load(open('data/MosquitoSound/MosquitoSound_TRAIN.arff', 'r'))
data = np.array(dataset['data'])
X_train, y_train = np.copy(data[:, :-1]), np.copy(data[:, -1])
# to free up RAM
del dataset
X_train = X_train.astype('float32')
dataset = arff.load(open('data/MosquitoSound/MosquitoSound_TEST.arff', 'r'))
data = np.array(dataset['data'])
X_test, y_test = np.copy(data[:, :-1]), np.copy(data[:, -1])
# to free up RAM
del dataset
X_test = X_test.astype('float32')

transfomer_GAF = GramianAngularField(image_size=64)
X_train_GAF = transfomer_GAF.fit_transform(X_train)
X_test_GAF = transfomer_GAF.fit_transform(X_test)
np.save('data/X_train_GAF_64.npy', X_train_GAF)
np.save('data/X_test_GAF_64.npy', X_test_GAF)
np.save('data/y_train.npy', y_train)
np.save('data/y_test.npy', y_test)
