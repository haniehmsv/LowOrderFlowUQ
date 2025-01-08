# LowOrderFlowUQ
A machine-learning tool to predict strongly disturbed aerodynamic states using sparse noisy pressure measurements. This code is capable of quantifying aleatoric (data) and epistemic (model) uncertainty. 

Follow the steps below in the specified order to execute the workflow:
1. Train the Autoencoder:
Run the file 'autoencoder.py' to train an autoencoder that extracts the underlying latent variables from the generated data. These latent variables represent the reduced-order features of the vorticity field and lift coefficient.
2. Train a Deterministic Network:
Execute 'deterministicPressureNetwork.py' to train a Multi-Layer Perceptron (MLP) network. This network maps sparse, clean surface pressure measurements to the extracted latent variables deterministically. The trained model will later be used for Gramian calculation.
3. Calculate Dominant Directions:
Run 'noise_in_dominant_direction.py' to identify and store the dominant directions of both the measurement and latent variable spaces at each time step. Perturb the measurements along the dominant eigenvector of the measurement space Gramian, C_x.
4. Train the Bayesian Neural Network Using MC Dropout:
Use 'probabilisticPressureNetwork.py' to train a Bayesian Neural Network. This model estimates the statistics of the latent variables by predicting the mean and covariance matrix of a multivariate normal distribution in the latent space. The training process minimizes the negative log-likelihood.
5. Flow Reconstruction and Uncertainty Quantification:
Finally, execute 'flowReconstructionAndUQ.py' to:
- Estimate the latent variables while quantifying aleatoric (data-driven) and epistemic (model-driven) uncertainties.
- Map the estimated latent space samples back to the original high-dimensional space to reconstruct the vorticity field and lift coefficient.

