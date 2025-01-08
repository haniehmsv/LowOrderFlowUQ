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

# Split train-test data
X_train, X_test, X_train_CL, X_test_CL, X_train_pres, X_test_pres = train_test_split(y_1, y_CL, sensors, test_size=0.2, random_state=42)

# Conv2D-MLP autoencoder
batch_size = 128
continue_state = False  # Set this to True to continue training from a saved model
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
    act = 'tanh'
    input_img = Input(shape=(120,240,1))
    x1 = Conv2D(32, (3,3),activation=act, padding='same')(input_img)
    x1 = Conv2D(32, (3,3),activation=act, padding='same')(x1)
    x1 = MaxPooling2D((2,2),padding='same')(x1)
    x1 = Conv2D(16, (3,3),activation=act, padding='same')(x1)
    x1 = Conv2D(16, (3,3),activation=act, padding='same')(x1)
    x1 = MaxPooling2D((2,2),padding='same')(x1)
    x1 = Conv2D(8, (3,3),activation=act, padding='same')(x1)
    x1 = Conv2D(8, (3,3),activation=act, padding='same')(x1)
    x1 = MaxPooling2D((5,5),padding='same')(x1)
    x1 = Conv2D(4, (3,3),activation=act, padding='same')(x1)
    x1 = Conv2D(4, (3,3),activation=act, padding='same')(x1)
    x1 = Reshape([12*6*4])(x1)
    x1 = Dense(256,activation=act)(x1)
    x1 = Dense(128,activation=act)(x1)
    x1 = Dense(64,activation=act)(x1)
    x1 = Dense(32,activation=act)(x1)
    x_lat = Dense(3,activation=act)(x1)

    ## lift augmentation
    x_CL = Dense(32,activation=act)(x_lat)
    x_CL = Dense(64,activation=act)(x_CL)
    x_CL = Dense(32,activation=act)(x_CL)
    x_CL_final = Dense(1)(x_CL)

    ## Upsampling connected to CNN
    x1 = Dense(32,activation=act)(x_lat)
    x1 = Dense(64,activation=act)(x1)
    x1 = Dense(128,activation=act)(x1)
    x1 = Dense(256,activation=act)(x1)
    x1 = Dense(288,activation=act)(x1)
    x1 = Reshape([6,12,4])(x1)
    x1 = Conv2D(4, (3,3),activation=act, padding='same')(x1)
    x1 = Conv2D(4, (3,3),activation=act, padding='same')(x1)
    x1 = UpSampling2D((5,5))(x1)
    x1 = Conv2D(8, (3,3),activation=act, padding='same')(x1)
    x1 = Conv2D(8, (3,3),activation=act, padding='same')(x1)
    x1 = UpSampling2D((2,2))(x1)
    x1 = Conv2D(16, (3,3),activation=act, padding='same')(x1)
    x1 = Conv2D(16, (3,3),activation=act, padding='same')(x1)
    x1 = UpSampling2D((2,2))(x1)
    x1 = Conv2D(32, (3,3),activation=act, padding='same')(x1)
    x1 = Conv2D(32, (3,3),activation=act, padding='same')(x1)
    x_final = Conv2D(1, (3,3),padding='same')(x1)

    ## training
    model = Model(input_img, [x_final,x_CL_final])
    model.compile(optimizer='adam', loss='mse',loss_weights=[1,0.05]) # beta = 0.05 determined by L-curve analysis
    
from keras.callbacks import ModelCheckpoint,EarlyStopping
model_cb=ModelCheckpoint('./autoencoder_model.keras', monitor='val_loss',save_best_only=True,verbose=1)
early_cb=EarlyStopping(monitor='val_loss', patience=200,verbose=1)
cb = [model_cb, early_cb]
history = model.fit(X_train,[X_train,X_train_CL],epochs=50000,batch_size=batch_size,verbose=1,callbacks=cb,shuffle=True,validation_data=(X_test, [X_test,X_test_CL]))

## saving history
df_results = pd.DataFrame(history.history)
df_results['epoch'] = history.epoch
df_results.to_csv(path_or_buf='./history.csv',index=False)

# saving learned latent variables
encoder = Model(inputs=model.input, outputs=model.get_layer('dense_4').output)
x_lat = encoder.predict(y_1)
np.save('x_lat.npy', x_lat)