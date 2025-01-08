# This file addresses calculation of measurement space Gramian matrix for the deterministic pressure network, 
# for the purpose of finding the dominant direction of measurements contributing to the latent space estimation.
import tensorflow as tf
from keras.layers import Input, Add, Dense, Conv2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D, UpSampling2D, Flatten, Reshape, LSTM, Concatenate, Conv2DTranspose, Dropout
from keras.models import Model
from keras import backend as K
from keras.regularizers import l2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm as tqdm
from scipy.io import loadmat
import pickle
import h5py
import matplotlib.pyplot as plt

# load 11 sparse pressure data stacked with their x and y positions: stored in tensor sensors of shape (n,33)

# load extracted latent variables from the autoencoder: x_lat of shape (n,3)
x_lat = np.load('x_lat.npy')

# load the deterministic pressure network trained in deterministicPressureNetwork.py
model = tf.keras.models.load_model('deterministicPressureNetwork_model.keras')

def input_prior(data, std_pres, std_loc=0.0):
    """
    Adds Gaussian noise to a vector of data.

    Args:
    - data: A numpy array of data (vector).
    - std_pres: The standard deviation of the Gaussian noise in pressure readings.
    - std_loc: The standard deviation of the Gaussian noise in sensor coordinates.

    Returns:
    - noisy_data: A numpy array with added Gaussian noise.
    """
    npr = data.shape[0]//3  # number of pressure sensors
    nloc = data.shape[0] - npr
    noise_pres = np.random.normal(0, std_pres, npr)
    noise_loc = np.random.normal(0, std_loc, nloc)
    noise = np.concatenate((noise_pres,noise_loc), axis=0)
    noisy_data = data + noise
    return noisy_data

class calculate_jacobian(tf.keras.Model):
    def __init__(self, net):
        super(calculate_jacobian, self).__init__()
        self.net = net

    def call(self, x):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            y = self.net(x, training=False)
        jac = tape.jacobian(y,x)
        return jac

def gramian(model, X, std_pres:float, std_loc:float, iterations:int=100):
    """
    Calculates the measurement space Gramian matrix for the deterministic pressure network.

    Args:
    - model: The deterministic pressure network.
    _ X: A numpy array of input data.
    - std_pres: The standard deviation of the Gaussian noise in pressure readings.
    - std_loc: The standard deviation of the Gaussian noise in sensor coordinates.
    _ iterations: The number of iterations to calculate the Gramian matrix using the Monte Carlo method.

    Returns:
    - Cx: The Gramian matrix in the measurement space.
    - Clat: The Gramian matrix in the latent space.
    """
    
    jac_model = calculate_jacobian(model)
    jacs_ensemble = []
    Cx = np.zeros((X.shape[0],X.shape[1],X.shape[1]))
    Clat = np.zeros((X.shape[0],3,3))
    
    for j in range(iterations):
        noisy_X = np.array([input_prior(x,std_pres,std_loc) for x in X])
        jac = jac_model(noisy_X)
        jacs_ensemble.append(tf.einsum('bxby->bxy', jac))
    jacs_ensemble = np.stack(jacs_ensemble)
    
    for i in range(iterations):
        transposed_tensor = tf.transpose(jacs_ensemble[i], perm=[0, 2, 1])
        Cx += tf.matmul(transposed_tensor, jacs_ensemble[i])*(std_pres**2)
        Clat += tf.matmul(jacs_ensemble[i], transposed_tensor)*(std_pres**2)
    Cx /= (iterations-1)
    Clat /= (iterations-1)
    return Cx, Clat

def calculate_dominant_modes(C_gramian, idx:np.ndarray, threshold:float=0.99):
    """
    Calculates the r_x dominant modes, denoted by U_r in our paper.

    Args:
    _ C_gramian: The Gramian matrix in either the measurement space or the latent space.
    - idx: The indices of the Gramian matrix to calculate the dominant modes. Used for parallel/chunk computation.
    - threshold: The threshold for the cumulative energy to determine the number of dominant modes.

    Returns:
    - num_modes_list: A list of the number of dominant modes for each index.
    - dominant_modes_list: A list of the dominant modes for each index.
    """
    dominant_modes_list = []
    num_modes_list = []
    for i in idx:
        U, S, Vt = np.linalg.svd(C_gramian[i,:,:], full_matrices=False)
        cumulative_energy = np.cumsum(S**2) / np.sum(S**2)
        num_modes = np.searchsorted(cumulative_energy, threshold) + 1
        dominant_modes = U[:, :num_modes]
        num_modes_list.append(num_modes)
        dominant_modes_list.append(dominant_modes)
    return num_modes_list, dominant_modes_list

def perturb_pressure(X, dominant_modes, std_pres:float, std_loc:float=0.0):
    """
    Adds Gaussian noise to a vector of data in the direction of dominant modes
    """
    noise_vector = np.zeros_like(X)
    n_mode = dominant_modes.shape[1]
    for i in range(n_mode):
        # Generate a random noise coefficient
        noise_pres = np.random.normal(0, std_pres)
        noise_loc = np.random.normal(0, std_loc)
        
        # Add noise in the direction of the dominant mode
        noise_vector += np.concatenate((noise_pres * dominant_modes[:11,i], noise_loc * dominant_modes[11:,i]), axis=0)
    noisy_data = X + noise_vector
    return noisy_data

def calculate_gramian(model_gramian, X, idx, std_pres, std_loc, iterations):
    # compute Cx and find dominant directions
    Cx, Clat = gramian(model_gramian, X, std_pres, std_loc, iterations=iterations)
    num_modes_x_list, dominant_modes_x_list = calculate_dominant_modes(Cx, idx)
    num_modes_lat_list, dominant_modes_lat_list = calculate_dominant_modes(Clat, idx)
    return dominant_modes_x_list, dominant_modes_lat_list

def calculate_noisy_pressure(model, X, idx, std_pres:float, std_loc:float=0.0, iterations_Cx=100):
    """Adds noise to the pressure measurements in the direction of the dominant modes."""
    # compute Cx and find dominant directions
    dominant_modes_x_list, dominant_modes_lat_list = calculate_gramian(model, X, idx, std_pres, std_loc, iterations_Cx)
    
    dominant_mode_lat = np.array([mode[:,0] for mode in dominant_modes_lat_list])
    dominant_mode_lat = np.vstack(dominant_mode_lat)
    dominant_modes_x = np.array([mode[:,0] for mode in dominant_modes_x_list])
    dominant_modes_x = np.vstack(dominant_modes_x)
    
    # perturb the pressure measurements
    X_noisy = np.array([perturb_pressure(X[i],dominant_modes_x_list[i],std_pres,std_loc) for i in range(len(X))])

    return X_noisy, dominant_mode_lat, dominant_modes_x



iterations_Cx = 100
std_pres = 5e-03    # 0.1% accuracy
std_loc = 0.0
chunk_size = 1000
dominant_mode_lat = []
dominant_mode_x = []
X_noisy = []
num_chunks = (len(sensors) + chunk_size - 1) // chunk_size
for i in range(num_chunks):
    start = i * chunk_size
    end = min(start + chunk_size, len(sensors))
    X_chunk = sensors[start:end]
    idx_chunk = indices[0:end-start]
    X_noisy_chunk, dominant_mode_lat_chunk, dominant_modes_x_chunk = calculate_noisy_pressure(model, X_chunk, idx_chunk, std_pres, std_loc, iterations_Cx=iterations_Cx)
    X_noisy.append(X_noisy_chunk)
    dominant_mode_lat.append(dominant_mode_lat_chunk)
    dominant_mode_x.append(dominant_modes_x_chunk)

dominant_mode_lat = np.concatenate(dominant_mode_lat, axis=0)
dominant_mode_x = np.concatenate(dominant_mode_x, axis=0)
X_noisy = np.concatenate(X_noisy, axis=0)

arrays = {
    'X_noisy': X_noisy,
    'dominant_mode_lat': dominant_mode_lat,
    'dominant_mode_x': dominant_mode_x,
}

with h5py.File('noise_in_dominant_direction.h5', 'w') as f:
    for name, array in arrays.items():
        f.create_dataset(name, data=array)
