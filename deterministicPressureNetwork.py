# A MLP network that maps pressure measurements stacked with the sensor coordinates to the latent variables of the autoencoder. The mapping is deterministic here, 
# used only for Gramian calculations. The network is trained to minimize the mean squared error between the predicted latent variables and the actual latent variables.
import tensorflow as tf
from keras.layers import Input, Add, Dense, Conv2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D, UpSampling2D, Flatten, Reshape, LSTM, Concatenate, Conv2DTranspose, Dropout
from keras.models import Model
from keras import backend as K
from keras.regularizers import l2
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm as tqdm
from scipy.io import loadmat
import pickle
import h5py
import matplotlib.pyplot as plt

# Load data: 
# vorticity field: 2D field of size (y,x)=(120,240) stored in tensor y_1 of shape (n,120,240,1)
# lift coefficient: 1D time series of length n stored in tensor y_CL of shape (n,1)
# 11 sparse pressure data stacked with their x and y positions: stored in tensor sensors of shape (n,33)

# load extracted latent variables from the autoencoder: x_lat of shape (n,3)
x_lat = np.load('x_lat.npy')

# Split train-test data
indices = np.arange(y_1.shape[0])
X_train, X_test, X_train_CL, X_test_CL, X_train_pres, X_test_pres, idx_train, idx_test = train_test_split(y_1, y_CL, sensors, indices, test_size=0.2, random_state=42)
Y_train_lat = x_lat[idx_train]
Y_test_lat = x_lat[idx_test]

# MLP deterministic pressure network
batch_size = 128
N = X_train.shape[0]
dropout_rate = 0.05
tau = 0.01
length_scale = 1e-2
reg = length_scale**2 * (1 - dropout_rate) / (2. * N * tau) # weight decay which here is calculated to be 1e-07
continue_state = False   # Set this to True to continue training from a saved model
if continue_state:
    # Load the previously saved model
    if os.path.exists('./model.keras'):
        print("Loading pre-trained model...")
        model = tf.keras.models.load_model('./model.keras')
    else:
        print("Pre-trained model not found. Starting fresh training...")
        continue_state = False
else:
    ## MLP for pressure
    act = 'relu'
    input_meas = Input(shape=(33,))
    x1 = Dense(64,activation=act, kernel_regularizer=l2(reg))(input_meas)
    x1 = Dropout(rate=dropout_rate)(x1, training=True)
    x1 = Dense(128,activation=act, kernel_regularizer=l2(reg))(x1)
    x1 = Dropout(rate=dropout_rate)(x1, training=True)
    x1 = Dense(256,activation=act, kernel_regularizer=l2(reg))(x1)
    x1 = Dropout(rate=dropout_rate)(x1, training=True)
    x1 = Dense(512,activation=act, kernel_regularizer=l2(reg))(x1)
    x1 = Dropout(rate=dropout_rate)(x1, training=True)
    x1 = Dense(256,activation=act, kernel_regularizer=l2(reg))(x1)
    x1 = Dropout(rate=dropout_rate)(x1, training=True)
    x1 = Dense(128,activation=act, kernel_regularizer=l2(reg))(x1)
    x1 = Dropout(rate=dropout_rate)(x1, training=True)
    x1 = Dense(64,activation=act, kernel_regularizer=l2(reg))(x1)
    x1 = Dropout(rate=dropout_rate)(x1, training=True)
    y_final = Dense(3, kernel_regularizer=l2(reg))(x1)

    ## training
    model = Model(input_meas, y_final)
    # opt = tf.keras.optimizers.Adam(amsgrad=True)
    model.compile(optimizer='adam', loss='mse')
    
from keras.callbacks import ModelCheckpoint,EarlyStopping
model_cb=ModelCheckpoint('./deterministicPressureNetwork_model.keras', monitor='val_loss',save_best_only=True,verbose=1)
early_cb=EarlyStopping(monitor='val_loss', patience=200,verbose=1)
cb = [model_cb, early_cb]
history = model.fit(X_train_pres,Y_train_lat,epochs=50000,batch_size=batch_size,verbose=1,callbacks=cb,shuffle=True,validation_data=(X_test_pres, Y_test_lat))

## saving history
df_results = pd.DataFrame(history.history)
df_results['epoch'] = history.epoch
df_results.to_csv(path_or_buf='./history.csv',index=False)

