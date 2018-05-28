import os
import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from keras.utils import np_utils
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import  Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.datasets import cifar10
from keras.callbacks import ModelCheckpoint

def make_model():
    k = float(np.random.rand()*1+0.2)
    print ('## k = %.3f' % k)
    winit1 = k/np.sqrt(5*5*1)
    winit2 = k/np.sqrt(5*5*64)
    winit3 = k/np.sqrt(5*5*128)


    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=5, padding='same',
                    data_format='channels_first',
                    use_bias=False, 
                    kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=winit1),
                    activation='relu', input_shape=(1,96,96)))
    model.add(MaxPooling2D(pool_size=2, strides=(2,2)))
    model.add(Conv2D(filters=128, kernel_size=5, padding='same',
                    data_format='channels_first',
                    use_bias=False,
                    kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=winit2),
                    activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=(2,2)))
    model.add(Conv2D(filters=256, kernel_size=5, padding='same',
                    data_format='channels_first',
                    use_bias=False,
                    kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=winit3),
                    activation='relu'))
    model.add(MaxPooling2D(pool_size=12, strides=(12,12)))
    model.add(Flatten())

    winitD1 = k/np.sqrt(4032)
    winitD2 = k/np.sqrt(300)
    model.add(Dropout(0.5))
    model.add(Dense(300, activation='relu',
                    kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05),
                    bias_initializer='Ones'))

    model.add(Dense(8, activation='softmax',
    kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=winitD2),
    bias_initializer='Zeros'))

    print(model.summary())
    
    # compile the model
    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-5, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,
             metrics=['accuracy'])

    return model

def make_cv_folds(X, y, folds, fold_num):
    X_train = []
    y_train = []
    X_valid = []
    y_valid = []

    for index, fold in enumerate(folds):
        if fold == fold_num:
            X_valid.append(X[index])
            y_valid.append(y[index])
        else:
            X_train.append(X[index])
            y_train.append(y[index])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_valid = np.array(X_valid)
    y_valid = np.array(y_valid)


    return X_train, y_train, X_valid, y_valid


if __name__ == '__main__':
    # Data loading
    dataset_path = 'save_data/npy_files'

    X = np.load(os.path.join(dataset_path,'X.npy'))
    y = np.load(os.path.join(dataset_path,'y.npy'))
    folds = np.load(os.path.join(dataset_path,'folds.npy'))
    
    # rescale [0,255] -> [0,2]    
    X = X.astype('float32')/255*2

    # one-hot encode the labels
    num_classes = len(np.unique(y))
    y = keras.utils.to_categorical(y, num_classes)
    
    

    
    # CV Training
    for times in range(10):
        for val_fold in range(10):
            X_train, y_train, X_valid, y_valid = make_cv_folds(X, y, folds, val_fold)
            
            print('X_train shape :', X_train.shape)
            print('y_train shape :', y_train.shape)
            
            
            # printing number of training, validation, and test images
            print(X_train.shape[0], 'train samples')
            #print(X_test.shape[0], 'test samples')
            print(X_valid.shape[0], 'validation samples')
            #X_test = X_test.astype('float32')/255
            model = make_model()
            filepath="result/{times:02d}/weights-improvement-{val_fold:02d}-{val_acc:.2f}.hdf5"
            checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5',
                                    monitor='val_acc',
                                    verbose=1, save_best_only=True)

            hist = model.fit(X_train, y_train, batch_size=64, epochs=50,
                        validation_data=(X_valid, y_valid),
                        callbacks=[checkpointer], verbose=2, shuffle=True)
  
        