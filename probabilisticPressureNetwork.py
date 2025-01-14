# Train a Bayesian NN to predict statistics of the latent variables using MC dropout
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

class NoisyDataGenerator(Sequence):
    """Augmenting input meausrements with Gaussian noise"""
    def __init__(self, X, Y, batch_size, std_pres, std_loc):
        self.X = X   # Clean input measurements
        self.Y = Y   # True output latent variables
        self.batch_size = batch_size
        self.std_pres = std_pres  # Standard deviation for pressure noise
        self.std_loc = std_loc    # Standard deviation for location noise
        self.indices = np.arange(len(X))
        self.on_epoch_end()

    def __len__(self):
        # Return the number of batches per epoch
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size

        # Ensure the end index does not exceed the size of X
        end_idx = min(end_idx, len(self.X))

        # Generate indices for the current batch
        batch_indices = self.indices[start_idx:end_idx]

        # Ensure batch_indices do not exceed the size of X
        if np.any(batch_indices >= len(self.X)):
            print(f"Warning: batch_indices out of bounds. Batch indices: {batch_indices}")
            batch_indices = batch_indices[batch_indices < len(self.X)]
        
        # Select the batch
        X_batch = self.X[batch_indices]
        Y_batch = self.Y[batch_indices]
        
        # Add noise to the pressure and location measurements in the batch
        noisy_X_batch = X_batch + np.concatenate((
            np.random.normal(0, self.std_pres, (X_batch.shape[0], 11)),  # Pressure noise
            np.random.normal(0, self.std_loc, (X_batch.shape[0], 22))    # Location noise
        ), axis=1)

        return noisy_X_batch, Y_batch

    def on_epoch_end(self):
        # Shuffle the dataset at the end of each epoch (optional)
        np.random.shuffle(self.indices)

def build_covariance_matrix(L_elements):
    """
    Constructs a covariance matrix from learned elements of a lower-triangular matrix L.
    
    Args:
        L_elemets: 6 learned elements of a lower-triangular matrix L.
    
    Returns:
        covariance_matrix: Tensor of shape (batch_size, 3, 3).
    """
    batch_size = tf.shape(L_elements)[0]
    L1 = tf.math.exp(0.5*L_elements[:, 0])
    L2 = tf.math.exp(0.5*L_elements[:, 1])
    L3 = tf.math.exp(0.5*L_elements[:, 2])
    L12 = L_elements[:, 3]
    L13 = L_elements[:, 4]
    L23 = L_elements[:, 5]
    
    # First row of L
    row1 = tf.stack([L1, tf.zeros_like(L1), tf.zeros_like(L1)], axis=1)

    # Second row of L
    row2 = tf.stack([L12, L2, tf.zeros_like(L2)], axis=1)

    # Third row of L
    row3 = tf.stack([L13, L23, L3], axis=1)
    
    L = tf.stack([row1, row2, row3], axis=1)
    covariance_matrix = tf.matmul(L, L, transpose_b=True)  # LL^T
    return covariance_matrix

def multivariate_gaussian_nll(y_true, y_pred):
    """
    Computes the negative log-likelihood for a multivariate Gaussian distribution.
    
    Args:
        y_true: Tensor of shape (batch_size, 3) containing true latent variables.
        y_pred: Tensor of shape (batch_size, 6) containing predicted means and log-variances/covariances.
    
    Returns:
        loss: Scalar tensor representing the average negative log-likelihood over the batch.
    """
    # Split y_pred into mean and log-variance/covariance
    mu = y_pred[:, :latent_dim]  # Shape: (batch_size, 3)
    L_elements = y_pred[:, latent_dim:]  # Shape: (batch_size, 6)
    
    # Build covariance matrix
    covariance_matrix = build_covariance_matrix(L_elements)  # Shape: (batch_size, 3, 3)
    
    # Compute the difference
    diff = tf.expand_dims(y_true - mu, axis=2)  # Shape: (batch_size, 3, 1)
    
    # Compute the inverse and determinant
    inv_cov = tf.linalg.inv(covariance_matrix)  # Shape: (batch_size, 3, 3)
    det_cov = tf.linalg.det(covariance_matrix)  # Shape: (batch_size,)
    det_cov = tf.maximum(det_cov, tf.keras.backend.epsilon())
    
    # Compute the quadratic term
    quadratic = tf.squeeze(tf.matmul(tf.matmul( tf.transpose(diff, perm=[0, 2, 1]), inv_cov), diff), axis=[1,2])  # Shape: (batch_size,)
    
    # Compute the NLL
    nll = 0.5 * (latent_dim * tf.math.log(2.0 * np.pi) + tf.math.log(det_cov) + quadratic)
    
    # Return the mean NLL over the batch
    return tf.reduce_mean(nll)

# MLP Bayesian pressure network with active dropout layers
batch_size = 128
N = X_train.shape[0]
dropout_rate = 0.05
tau = 0.01
length_scale = 1e-2
reg = length_scale**2 * (1 - dropout_rate) / (2. * N * tau) # weight decay which here is calculated to be 1e-07
latent_dim = 3

## Initialize the generator with your dataset
std_pres = 5e-03    # 0.1% accuracy
std_loc = 0.0
train_generator = NoisyDataGenerator(X_train_pres, Y_train_lat, batch_size, std_pres, std_loc)
test_generator = NoisyDataGenerator(X_test_pres, Y_test_lat, batch_size, std_pres, std_loc)

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
    y_mean = Dense(latent_dim, activation='tanh', kernel_regularizer=l2(reg))(x1)   # mean of the latent variables
    y_logvar = Dense(latent_dim * (latent_dim + 1) // 2, kernel_regularizer=l2(reg))(x1)    # 6 elements of a lower-triangular matrix
    y_final = Concatenate(name="final_output")([y_mean, y_logvar])

    ## training
    model = Model(input_meas, y_final)
    opt = tf.keras.optimizers.Adam(amsgrad=True)
    model.compile(optimizer=opt, loss=multivariate_gaussian_nll)
    
from keras.callbacks import ModelCheckpoint,EarlyStopping
model_cb=ModelCheckpoint('./probabilistic_model.keras', monitor='val_loss',save_best_only=True,verbose=1)
early_cb=EarlyStopping(monitor='val_loss', patience=500,verbose=1)
cb = [model_cb, early_cb]
history = model.fit(train_generator,epochs=50000,verbose=1,callbacks=cb,shuffle=True,validation_data=test_generator)

## saving history
df_results = pd.DataFrame(history.history)
df_results['epoch'] = history.epoch
df_results.to_csv(path_or_buf='./history.csv',index=False)