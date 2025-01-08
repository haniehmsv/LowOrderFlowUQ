# Now perturb the measurements in the most dominant directions and predict the learned parameters of a multi-variate Gaussian distribution in latent space
# using the trained network model in probabilisticPressureNetwork.py.
# The pre-trained decoder in autoencoder.py is used to reconstruct the vorticity field and lift coefficient samples together with their uncertainties.
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
from probabilisticPressureNetwork import build_covariance_matrix, multivariate_gaussian_nll

# log-likelihood and RMSE error functions for evaluation of model performance
def error(Y_true, Y_mean, Y_std):
    Y_true = tf.convert_to_tensor(Y_true, dtype=tf.float32)
    Y_mean = tf.convert_to_tensor(Y_mean, dtype=tf.float32)
    Y_std = tf.convert_to_tensor(Y_std, dtype=tf.float32)
    log_2pi = tf.constant(np.log(2 * np.pi), dtype=tf.float32)
    loglike = -0.5/(Y_std**2.) * tf.square(Y_mean - Y_true) - tf.math.log(Y_std) - 0.5*log_2pi
    if len(Y_true.shape)==4:    # for vorticity field
        loglike = tf.reduce_sum(loglike, axis=[1,2,3])
        loglike_per_pixel = loglike/(Y_true.shape[1]*Y_true.shape[2]*Y_true.shape[3])
    elif len(Y_true.shape)==3:  # for lift coefficient
        loglike = tf.reduce_sum(loglike, axis=[0,1,2])
        loglike_per_pixel = loglike/(Y_true.shape[0]*Y_true.shape[1]*Y_true.shape[2])
    
    rmse = tf.sqrt(tf.reduce_mean(tf.square(Y_mean - Y_true)))
    return loglike, loglike_per_pixel, rmse

def snapshot_at_maximum_uncertainty(S):
    """S: eigenvalues of the covariance matrix, derived from taking SVD"""
    std_difference = np.linalg.norm(S, axis=1)
    std_max_idx = np.argmax(std_difference)
    return std_max_idx

AoA = np.array([20,30,40,50,60])    # angles of attack
nsnap = 745   # number of snapshots per angle of attack per gust/base case
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

# load models
model = tf.keras.models.load_model('probabilistic_model.keras', custom_objects={'build_covariance_matrix': build_covariance_matrix, 'multivariate_gaussian_nll': multivariate_gaussian_nll})
model_decod = tf.keras.models.load_model('autoencoder_model.keras')
decoder = Model(inputs=model_decod.get_layer('dense_4').output, outputs=model_decod.get_layer('conv2d_16').output)
decoder_CL = Model(inputs=model_decod.get_layer('dense_4').output, outputs=model_decod.get_layer('dense_8').output)

# Perturb the measurements in the most dominant directions
file_name = 'noise_in_dominant_direction.h5'
dataset_names = [
    'X_noisy',
    'dominant_mode_lat',
    'dominant_mode_x',
]
stacked_data = {name: [] for name in dataset_names}
with h5py.File(file_name, 'r') as f:
    for name in dataset_names:
        # Append data to the corresponding list in the dictionary
        stacked_data[name].append(f[name][:])
            
for name in dataset_names:
    stacked_data[name] = np.concatenate(stacked_data[name], axis=0)

X_noisy = stacked_data['X_noisy']
X_noisy_train = X_noisy[idx_train]
X_noisy_test = X_noisy[idx_test]
dominant_mode_x = stacked_data['dominant_mode_x']

# 100 forward passes to generate samples of the statistics
MC_samples = np.array([model(X_noisy, training=True) for _ in range(100)])
# Epistemic
x_lat_mean_samples = MC_samples[:, :, :3]
x_lat_mean = np.mean(x_lat_mean_samples,axis=0)
x_lat_epistemic_std = np.std(x_lat_mean_samples, 0)
#Aleatoric
log_output = np.mean(MC_samples[:, :, 3:], 0)
x_lat_aleatoric_var = build_covariance_matrix(log_output)

# Drawing samples from the learned aleatoric distribution
n_samples = 100
# np.random.seed(42)
x_lat_aleatoric_samples = []
for i in range(x_lat_mean.shape[0]):
    mu = x_lat_mean[i]
    cov = x_lat_aleatoric_var[i]
    samples = np.random.multivariate_normal(mu, cov, size=n_samples)
    x_lat_aleatoric_samples.append(samples)
x_lat_aleatoric_samples = np.array(x_lat_aleatoric_samples)
x_lat_aleatoric_samples = np.transpose(x_lat_aleatoric_samples, (1, 0, 2))
x_lat_aleatoric_std = np.std(x_lat_aleatoric_samples, axis=0)

# Samples from the learned epistemic distribution
x_lat_epistemic_samples = x_lat_mean_samples
x_lat_epistemic_var = np.zeros((x_lat.shape[0], latent_dim, latent_dim))
for i in range(x_lat.shape[0]):
    snapshot_samples = x_lat_epistemic_samples[:, i, :]  # shape (n_samples, 3)
    x_lat_epistemic_var[i] = np.cov(snapshot_samples, rowvar=False)

## Reconstruction of the vorticity field and lift coefficient samples together with their uncertainties
# (done for only epistemic uncertainty as an example, can be extended to aleatoric uncertainty)
S, U, V = tf.linalg.svd(x_lat_epistemic_var)
std_max_idx = np.zeros((len(AoA)), dtype=int)
for i in range(len(AoA)):
    std_max_idx[i] = snapshot_at_maximum_uncertainty(S[i*nsnap:(i+1)*nsnap-245]) + i*nsnap
### Vorticity field
vort_epistemic_std_max = np.array([decoder(x[std_max_idx]) for x in x_lat_epistemic_samples])
vort_epistemic_std_max_mean = np.mean(vort_epistemic_std_max, axis=0)
vort_epistemic_std_max_std = np.std(vort_epistemic_std_max, axis=0)
### Lift coefficient
CL_samples = np.array([decoder_CL(x) for x in x_lat_epistemic_samples])
CL_mean = np.mean(CL_samples, axis=0)
CL_std = np.std(CL_samples, axis=0)