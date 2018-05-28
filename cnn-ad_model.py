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




if __name__ == '__main__':
    # Data loading
    X = np.load('data_scripts/ck_plus_data/npy_files/X.npy')
    y = np.load('data_scripts/ck_plus_data/npy_files/y.npy')


    # K-folds CV
    kf = KFold(n_splits=10, random_state=20)

    # fig = plt.figure(figsize=(20,5))
    # for i in range(36):
    #     ax = fig.add_subplot(3, 12, i+1, xticks=[], yticks=[])
    #     ax.imshow(np.squeeze(X_train[i]))
        
    # rescale [0,255] -> [0,2]    
    X = X.astype('float32')/255*2

    # one-hot encode the labels
    num_classes = len(np.unique(y))
    y = keras.utils.to_categorical(y, num_classes)
    #y_test = keras.utils.to_categorical(y_test, num_classes)

    # break training set into training and validation sets
    #(X_train, X_valid) = X_train[200:], X_train[:200]
    #(y_train, y_valid) = y_train[200:], y_train[:200]

    # train the model
    checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5',
                                verbose=1, save_best_only=True)
    # hist = model.fit(X_train, y_train, batch_size=64, epochs=1000,
    #                 validation_data=(X_valid, y_valid),
    #                 callbacks=[checkpointer], verbose=2, shuffle=True)

    # CV Training
    for train_index, test_index in kf.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)

        X_train, X_valid = X[train_index], X[test_index]
        y_train, y_valid = y[train_index], y[test_index]
        # print shape of training set
        print('x_train shape :', X_train.shape)
        
        
        # printing number of training, validation, and test images
        print(X_train.shape[0], 'train samples')
        #print(X_test.shape[0], 'test samples')
        print(X_valid.shape[0], 'validation samples')
        #X_test = X_test.astype('float32')/255
        model = make_model()

        datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)
        datagen_val = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)
        datagen_val.fit(X_valid)
        datagen.fit(X_train)
        hist = model.fit_generator(datagen.flow(X_train, y_train, batch_size=64),
                            steps_per_epoch=len(X_train) / 64, epochs=100,
                            validation_data=(datagen_val.flow(X_valid, y_valid)),
                            use_multiprocessing=True,
                            callbacks=[checkpointer], verbose=2, shuffle=True)
        # hist = model.fit(X_train, y_train, batch_size=64, epochs=100,
        #             validation_data=(X_valid, y_valid),
        #             callbacks=[checkpointer], verbose=2, shuffle=True)
        