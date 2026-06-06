# BCAT on 2d Spherical Shallow Water System

## What is Block Causal Transformer (BCAT)?
BCAT (https://arxiv.org/abs/2501.18972) is a PDE foundational model for the 2d fluid systems, including compressible/incompressible Navier-Stokes equations and shallow water equations.  BCAT reduces the computational costs by using block tokens instead of tokenizing the functions at all spacetime points.  

## What we did?
While the block is inherently 2d Cartesian, some systems with spherical symmetry are more naturally described in polar coordinates and can be challenging to BCAT.  This project applies BCAT on 2d spherical shallow water system (https://en.wikipedia.org/wiki/Shallow_water_equations)
$$ x $$

We use the PDE Bench's dataset () and studies its performance and interprete the model's state parameters in various cases.  More specifically, we did the followings:

(1) tuned the model hyperparameters to find the optimal range of convergence for each block size,  
(2) investigated the effect of "block sizes" (block sizes in {8, 16, 32, 64, 128}) with the following metrics: The error images of rollout predictions, mean square errors over rollout horizon, conservation of mass (dM/dt),
(3) studied the effect of noise, relevance of choice of initial frame for the rollout prediction and temporal attention,
(4) implemented a physics informed neural network (PINN) by constraining dM/dt to be small.

## Each file
**Core files:**
- all_dataset.py: The ShallowWater2D class organizes PDE Bench's shallow water dataset (1000 samples, 128*128 spatial resolutions, 101 time steps, 1 channel) in the .h5 format into a Python list, d = {"data_mask", "data", "t", "symbol_input"} after 
(1) shuffling the samples depending on user preference (train/test), 
(2) limiting the number of samples to accommodate limited RAM space, 
(3) specifying the unit of time step, 
(4) augmenting user-specified noise.

- embedder.py: (1) encodes (downsamples) the height of water at each spacetime h(x,y;t) within a block into a block token in the latent space using Conv2d, (2) decodes (upsamples) the next-frame prediction for the block tokens using ConvTranspose2d.

- bcat.py: Uses block caual mask that block outs the influence from future tokens in the Pytorch's built-in TransformerEncoderLayers, which implements self-attention (token updates) and feed-forward (next frame token prediction).

- rollout.py: Based on the user specified initial frame, the trained model autogregressively predicts/decodes all block tokens at the next time frame, append it to the new input data (data_all) and continue predicting until it predicts 101 time steps (the max time step used in training).

**Files for performance metric:**
- conservation.py: compute the mass and time derivative of mass of the ground truth and roll-out predictions over the prediction horizon and plot the graph comparing the two.

- initial_time.py: plots the root-mean-square-errors of the rollout prediction with different initial time frames and compare them.

- rollout_error.py: plots the mean-square error plots over the rollout prediction horizon.

**PINN:**
- main_PINN.py: use the training loss = mean square error + w_PINN * (dM/dt)^2 to provide a soft constraint correspondint to the conservation law.

**Training and rollout prediction main files:**
- main2.py: 
(1) Using "params = types.SimpleNameSpace" (a kind of dictionary), we specify the dataset related parameters, training preferences and model parameters (e.g. dim = embedding dimension).
(2) Using PyTorch's utils library, we create two ShallowWater2D objects, one for training and the other for validation, and pass them to two DataLoaders.
(3) We construct the model (a BCAT object), optimizer (AdamW)/
(4) We normalize the dataset and enter the training loop.  Once the training is over, we save
  - the trained state vectors (checkpoint) to "bcat_shallow_water.pt",
  - the training/validation losses to 'training_validation_losses.csv',
  - the loss per epoch graph to 'training_validation_loss_plot.pdf'
  and print out the number of model and encoder/decoder parameters.
(5) We load the checkpoint, generate rollout predictions and plot the predictions vs ground truth vs errors for 4 time steps.
(6) We save the performance metric and model hyperparameters.

- main_few_shots.py: loads the checkpoint files (state vector of trained model) and enters a few epoch training loop.  Then performs rollout prediction as in main2.py.

- main_zero_shot.py: loads the checkpoint files (state vector of trained model) and performs rollout prediction as in main2.py.


## How to use 
- For the first time (training) use: adjust the parameters in "params" in main2.py.
- Few shot learning after training: adjust the parameters in "params" in main.few_shots.py to those of trained model and load the checkpoint file.
- Zero-shot learning after training: adjust the parameters in "params" in main.few_shots.py to those of trained model and load the checkpoint file.
- PINN: adjust w_PINN in the main_PINN.py.
