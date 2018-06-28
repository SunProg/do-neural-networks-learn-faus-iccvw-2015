import os
import keras
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers.merge import concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import  Input, Dense, Conv2D, Dropout, Flatten, MaxPooling2D, LSTM
from keras.datasets import cifar10
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json


def make_model():
    k = float(np.random.rand()*1+0.2)
    print ('## k = %.3f' % k)
    winit1 = k/np.sqrt(5*5*1)
    winit2 = k/np.sqrt(5*5*64)
    winit3 = k/np.sqrt(5*5*128)
    
    cnn_input = Input(shape=(1,96,96))
    conv1 = Conv2D(filters=64, kernel_size=5, padding='same',
                    data_format='channels_first',
                    use_bias=False, 
                    kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=winit1),
                    activation='relu')(cnn_input)
    maxpool1 = MaxPooling2D(pool_size=2, strides=(2,2))(conv1)
    
    conv2 = Conv2D(filters=128, kernel_size=5, padding='same',
                    data_format='channels_first',
                    use_bias=False,
                    kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=winit2),
                    activation='relu')(maxpool1)
    maxpool2 = MaxPooling2D(pool_size=2, strides=(2,2))(conv2)
    
    conv3 = Conv2D(filters=256, kernel_size=5, padding='same',
                    data_format='channels_first',
                    use_bias=False,
                    kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=winit3),
                    activation='relu')(maxpool2)
    maxpool3 = MaxPooling2D(pool_size=12, strides=(12,12))(conv3)
    flatten = Flatten()(maxpool3)

    winitD1 = k/np.sqrt(4032)
    winitD2 = k/np.sqrt(300)
    dropout = Dropout(0.5)(flatten)
    dense1 = Dense(300, activation='relu',
                    kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05),
                    bias_initializer='Ones')(dropout)

    cnn_output = Dense(8, activation='softmax',
    kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=winitD2),
    bias_initializer='Zeros')(dense1)

    cnn_model = Model(inputs=cnn_input, outputs=cnn_output)
    print(cnn_model.summary())
    
    # compile the model
    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-5, momentum=0.9)
    cnn_model.compile(loss='categorical_crossentropy', optimizer=sgd,
             metrics=['accuracy'])
    
    # LSTM Model
    rnn_input = Input(shape=(15, 64))
    lstm1 = LSTM(250, return_sequences=True)(rnn_input)
    feature_and_class = Input(shape=(15,4040,))
    merge = concatenate([lstm1, feature_and_class])
    lstm2 = LSTM(250, return_sequences=True)(merge)
    rnn_output = Dense(64, activation='softmax')(lstm2)
    
    rnn_model = Model(inputs=[rnn_input, feature_and_class], outputs=rnn_output)
    print(rnn_model.summary())
    rnn_model.compile(loss='categorical_crossentropy', optimizer=sgd,
             metrics=['accuracy'])
    
    return cnn_model, rnn_model

def load_data(dataset_path='./npy_files'):
    X = np.load(os.path.join(dataset_path,'X.npy'))
    y = np.load(os.path.join(dataset_path,'y.npy'))
    folds = np.load(os.path.join(dataset_path,'folds.npy'))
    feature_class = np.load(os.path.join(dataset_path, 'feature_class.npy'))
    facs = np.load(os.path.join(dataset_path,'np_facs.npy'))

    # rescale [0,255] -> [0,2]    
    X = X.astype('float32')/255*2

    # one-hot encode the labels
    num_classes = len(np.unique(y))
    y = keras.utils.to_categorical(y, num_classes)

    return X, y, folds, feature_class, facs


def make_facs_y(facs):
    facs_y = []
    for inst in facs:
        temp = list(inst)[:-1]
        temp.append(list(np.zeros(64)))
        facs_y.append(temp)
    facs_y = np.array(facs_y)
    
    return facs_y

def make_cv_folds(X, y, feature_class, facs, folds, fold_num):
    X_train = []
    y_train = []
    feature_class_train = []
    facs_train = []
    
    X_valid = []
    y_valid = []
    feature_class_valid = []
    facs_valid = []
    
    for index, fold in enumerate(folds):
        if fold == fold_num:
            X_valid.append(X[index])
            y_valid.append(y[index])
            feature_class_valid.append(feature_class[index])
            facs_valid.append(facs[index])
        else:
            X_train.append(X[index])
            y_train.append(y[index])
            feature_class_train.append(feature_class[index])
            facs_train.append(facs[index])
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_valid = np.array(X_valid)
    y_valid = np.array(y_valid)
    
    feature_class_train = np.array(feature_class_train)
    feature_class_valid = np.array(feature_class_valid)

    facs_train = np.array(facs_train)
    facs_valid = np.array(facs_valid)
    facs_y_train = make_facs_y(facs_train)
    facs_y_valid = make_facs_y(facs_valid)

    return (X_train, y_train, 
            X_valid, y_valid,
            feature_class_train, feature_class_valid,
            facs_train, facs_valid,
            facs_y_train, facs_y_valid
           )


def load_cnn_model(model_path='./result/'):
    with open(os.path.join(model_path, 'model_architecture.json'), 'r') as f:
        model = model_from_json(f.read())

    model.load_weights(os.path.join(model_path, 'weights-42-0.90.hdf5'))
    
    return model


X, y, folds, feature_class, facs = load_data('./save_data/npy_files/')

_, rnn_model = make_model()
cnn_model = load_cnn_model('./result/test/')

X_train, y_train, X_valid, y_valid, feature_class_train, feature_class_valid, facs_train, facs_valid, facs_y_train, facs_y_valid = make_cv_folds(X, y, feature_class, facs, folds, 0)

result_path_folder = './result/rnn_model'
filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
filepath = os.path.join(result_path_folder,filepath)
checkpointer = ModelCheckpoint(filepath=filepath,
                        monitor='val_acc',
                        verbose=1, save_best_only=True)
 
hist = rnn_model.fit([facs_train, feature_class_train ], facs_y_train, batch_size=64, epochs=100,
            validation_data=([facs_valid, feature_class_valid], facs_y_valid),
            callbacks=[checkpointer], verbose=2, shuffle=True)