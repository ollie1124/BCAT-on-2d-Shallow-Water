import sys
print(sys.executable)

import torch
from torch.utils.data import DataLoader
from all_datasets import ShallowWater2D
from embedder import PatchEmbedder
from bcat import BCAT
from rollout import rollout_model

from torch.utils.data import Subset

import os
import urllib.request
from types import SimpleNamespace
from itertools import islice

import pandas as pd
import csv

# -----------------------------
# 1. Download the shallow-water H5 file
# -----------------------------
# The url and name are from https://github.com/pdebench/PDEBench/blob/main/pdebench/data_download/README.md
# download_url = "https://darus.uni-stuttgart.de/api/access/datafile/133017" # diffusion reaction
# local_filename = "2d_reacdiff"
download_url = "https://darus.uni-stuttgart.de/api/access/datafile/133021" # shallow water
local_filename = "2D_rdb_NA_NA.h5"
local_path = os.path.join("", local_filename)
# local_path = os.path.join("/content", local_filename)

if not os.path.exists(local_path):
    print(f"Downloading dataset to {local_path} ...")
    urllib.request.urlretrieve(download_url, local_path)
    print("Download complete.")
else:
    print(f"Dataset already exists at {local_path}")




# ==============================
# Setting the parameters.
# ==============================
import types

params = types.SimpleNamespace(
    # Seolhwa: types.SimpleNamespace is a dictionary that allows dot access. Instead of params["data"]["x_num"], we can simply write params.data.x_num
    # This is used because the original project uses OmegaConf and Hydra which produce objects accessed like "config.data.x_num".
    # We kept it because we used alldataset.py to retrieve the data set from .h5 file, which relies on this dot access.
    data=types.SimpleNamespace(
        # This first part is the parameters for shallw water system dataset.
        shallow_water=types.SimpleNamespace(
            data_path=local_path,
            x_num=128,
            t_step=4
        ),

        # We can define other systems too.
          # e.g. diff_reaction=types.SimpleNamespace(...),

        # Below are default values:
        x_num=128,# This is for shallow water dataset, but for other dataset, this number might change.
        t_num=101,#26,#101,# Other systems (e.g. NS) have different time steps... (Reduced from 101 for performance)
        random_start=types.SimpleNamespace(
            train=False, 
            val=True, 
            test=False, 
            start_max=50 # change the start_max to 0 to disable random start, or set to a positive number to enable random start up to that number of time steps. This is used to test the effect of random start on the model performance.
            ),
        # random_start={'train': False, 'val': True, 'test': False, 'start_max': 0},
        # random_start={'test': types.SimpleNamespace(train=True, val=True, test=True, start_max=50), 'train': types.SimpleNamespace(train=True, val=True, test=True, start_max=50), 'val': types.SimpleNamespace(train=True, val=True, test=True, start_max=50)},
        train_val_test_ratio=[0.8, 0.1, 0.1],
        tie_fields=True,
        max_output_dimension=1
    ),
    overfit_test=False,
    noise=0,# from 0 to 1
    noise_type="additive", # "multiplicative",
    flip=False,
    rotate=False,
    use_raw_time=True, # Changed to True to include time data
    symbol=types.SimpleNamespace(symbol_input=False),
    num_workers=0,
    num_workers_eval=0,
    local_rank=0,
    n_gpu_per_node=1,
    global_rank=0,
    test_seed=42,
    batch_size=4, # Reduced to 2 as per user's current setting

    # Latent space encoding/decoding params
    dim = 128, # originally 1024, embedding dimension, a.k.a. hidden dimension of attention, latent space dimension, etc. (Increased from 64)
    x_num = 128,
    patch_size = 128,# originally 128 (64 for fine tuning)
    patch_num = 128//128, # params.x_num// params.patch_size,
    patch_num_output = 128//128, #params.patch_num, # in case the decoder (de)compresses the result
    conv_dim = 32,#Default: self.dim // 4 # decoder's inner neural network dimension.  embeding dim -> conv_dim -> patch_size

    # Transformer hyperparameters
    # These are in Table 9 of the BCAT paper.  I downsized the parameters.
    nhead = 2, # 8, originally. (Increased from 2)
    num_layers=2,# 12, originally (Increased from 2)
    dim_feedforward=128, # 275, originally (Increased from 64)
    max_len=128,#128,
    max_time_len =101, # Must match params.data.t_num (Reduced from 101 for performance)
    dropout=0

)

# Create a dummy symbol_env object
symbol_env = types.SimpleNamespace()


print(f"data_path has been set to: {params.data.shallow_water.data_path}")


# ===============================
# parameters with zero noise
# ===============================
params_zero_noise = types.SimpleNamespace(
    # Seolhwa: types.SimpleNamespace is a dictionary that allows dot access. Instead of params["data"]["x_num"], we can simply write params.data.x_num
    # This is used because the original project uses OmegaConf and Hydra which produce objects accessed like "config.data.x_num".
    # We kept it because we used alldataset.py to retrieve the data set from .h5 file, which relies on this dot access.
    data=types.SimpleNamespace(
        # This first part is the parameters for shallw water system dataset.
        shallow_water=types.SimpleNamespace(
            data_path=local_path,
            x_num=128,
            t_step=4
        ),

        # We can define other systems too.
          # e.g. diff_reaction=types.SimpleNamespace(...),

        # Below are default values:
        x_num=128,# This is for shallow water dataset, but for other dataset, this number might change.
        t_num=101,#26,#101,# Other systems (e.g. NS) have different time steps... (Reduced from 101 for performance)
        random_start=types.SimpleNamespace(
            train=False, 
            val=True, 
            test=False, 
            start_max=50 # change the start_max to 0 to disable random start, or set to a positive number to enable random start up to that number of time steps. This is used to test the effect of random start on the model performance.
            ),
        # random_start={'train': False, 'val': True, 'test': False, 'start_max': 0},
        # random_start={'test': types.SimpleNamespace(train=True, val=True, test=True, start_max=50), 'train': types.SimpleNamespace(train=True, val=True, test=True, start_max=50), 'val': types.SimpleNamespace(train=True, val=True, test=True, start_max=50)},
        train_val_test_ratio=[0.8, 0.1, 0.1],
        tie_fields=True,
        max_output_dimension=1
    ),
    overfit_test=False,
    noise=0,# from 0 to 1
    noise_type="additive", # "multiplicative",
    flip=False,
    rotate=False,
    use_raw_time=True, # Changed to True to include time data
    symbol=types.SimpleNamespace(symbol_input=False),
    num_workers=0,
    num_workers_eval=0,
    local_rank=0,
    n_gpu_per_node=1,
    global_rank=0,
    test_seed=42,
    batch_size=4, # Reduced to 2 as per user's current setting

    # Latent space encoding/decoding params
    dim = 128, # originally 1024, embedding dimension, a.k.a. hidden dimension of attention, latent space dimension, etc. (Increased from 64)
    x_num = 128,
    patch_size = 128,# originally 128 (64 for fine tuning)
    patch_num = 128//128, # params.x_num// params.patch_size,
    patch_num_output = 128//128, #params.patch_num, # in case the decoder (de)compresses the result
    conv_dim = 32,#Default: self.dim // 4 # decoder's inner neural network dimension.  embeding dim -> conv_dim -> patch_size

    # Transformer hyperparameters
    # These are in Table 9 of the BCAT paper.  I downsized the parameters.
    nhead = 2, # 8, originally. (Increased from 2)
    num_layers=2,# 12, originally (Increased from 2)
    dim_feedforward=128, # 275, originally (Increased from 64)
    max_len=128,#128,
    max_time_len =101, # Must match params.data.t_num (Reduced from 101 for performance)
    dropout=0

)













# ==============================
# 2. Initialize the ShallowWater2D dataset and DataLoader
# ==============================

import torch.utils.data as data

# Instantiate the ShallowWater2D dataset
shallow_water_dataset = ShallowWater2D(params, symbol_env=None, split="train", train=True)


# Create a DataLoader to stream the data
dataset_loader = data.DataLoader(
    shallow_water_dataset,
    batch_size=params.batch_size,  # Use params.batch_size
    shuffle=False,
    # shuffle=True, # Shuffle handled within the dataset for training
    num_workers=params.num_workers
)


import torch
from torch.utils.data import DataLoader

# create dataset
train_dataset = ShallowWater2D(
    params=params,
    symbol_env=None,
    split="train",
    train=True,
)

val_dataset = ShallowWater2D(
    params=params,
    symbol_env=None,
    split="val",
    train=False,
)

val_dataset_zero_noise = ShallowWater2D(
    params=params_zero_noise,
    symbol_env=None,
    split="val",
    train=False,
)




# dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=params.batch_size,
    num_workers=params.num_workers,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=params.batch_size,
    num_workers=params.num_workers_eval,
)




# ==============================
# Retrieving trained parameters 
# ==============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BCAT(params).to(device)

# # Load the trained model checkpoint
# checkpoint_path = "bcat_shallow_water.pt"
# checkpoint = torch.load(checkpoint_path, map_location=device)
# model.load_state_dict(checkpoint["model_state_dict"])
# model.to(device)
# model.eval() # Set model to evaluation mode
# print("Model loaded and set to evaluation mode.")

checkpoint_path = "bcat_shallow_water.pt"

try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Checkpoint loaded. Ready to rollout.")
except FileNotFoundError:
    print(f"Checkpoint file not found at {checkpoint_path}. ")
except RuntimeError as e:
    print(f"Error loading checkpoint due to model parameter shape mismatch: {e}. (model not loaded).")
except Exception as d:
    print(f"Error loading checkpoint: {d}.")
# finally:
#     print(f"Starting training from epoch {start_epoch}.")



print("Calculating data normalization stats...")
all_train_data = []
for i, b in enumerate(train_loader):
    all_train_data.append(b["data"])
    if i * params.batch_size > 100: # Use first 100 samples to estimate
        break

all_train_data = torch.cat(all_train_data, dim=0)
data_mean = all_train_data.mean().item()
data_std = all_train_data.std().item()

print(f"Calculated data mean: {data_mean:.4f}, data std: {data_std:.4f}")



# ================================================
# --- Code to execute rollout and plot results ---
# ================================================
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import DataLoader # Needed for val_loader re-creation if not already imported


# Get one batch from the validation loader for rollout
# Re-create val_loader to ensure it's fresh if it was exhausted
val_loader = DataLoader(
    val_dataset,
    batch_size=params.batch_size,
    num_workers=params.num_workers_eval
    #shuffle=True # For consistent validation results
)

val_loader_zero_noise = DataLoader(
    val_dataset_zero_noise,
    batch_size=params_zero_noise.batch_size,
    num_workers=params_zero_noise.num_workers_eval
    #shuffle=True # For consistent validation results
)

# Get a single batch for testing rollout
val_batch = next(iter(val_loader))
val_batch_zero_noise = next(iter(val_loader_zero_noise))

# Prepare inputs for rollout_model
# We take the first sample from the batch
data_sample = val_batch["data"][0:1, :, :, :, :].to(device) # (1, T, H, W, C)
data_sample_zero_noise = val_batch_zero_noise["data"][0:1, :, :, :, :].to(device) # (1, T, H, W, C)
times_sample = val_batch["t"][0:1, :].to(device) # (1, T)

# The input to rollout_model should be the first timestep of the data_sample
initial_time = 1
initial_input = data_sample[:, initial_time:initial_time+1, :, :, :]
initial_input_zero_noise = data_sample_zero_noise[:, initial_time:initial_time+1, :, :, :]
# initial_input = data_sample[:, :1, :, :, :]

# The full time sequence for the prediction horizon
# full_times_sequence = times_sample
full_times_sequence = times_sample[:,initial_time:]

# input_len is 1 because we are providing only the first frame as input
input_len = 1

# data_mask: a mask of ones if no specific channels are to be masked out
data_mask = torch.ones_like(initial_input[:, 0:1, :, :, :]).to(device)

print(f"Initial input shape for rollout: {initial_input.shape}")
print(f"Full times sequence shape for rollout: {full_times_sequence.shape}")

# Perform rollout prediction
with torch.no_grad():
    predicted_sequence = rollout_model(model, initial_input, full_times_sequence, input_len, data_mask, data_mean, data_std)

print(f"Predicted sequence shape: {predicted_sequence.shape}")




# Compare with ground truth
ground_truth_sequence = data_sample[:, initial_time + input_len:, :, :, :]
ground_truth_sequence_zero_noise = data_sample_zero_noise[:, initial_time + input_len:, :, :, :]
error = predicted_sequence - ground_truth_sequence
error_zero_noise = predicted_sequence - ground_truth_sequence_zero_noise

# Calculate and print Mean Squared Error
mse = torch.mean(error**2).item()
print(f"\n Rollout Mean Squared Error (MSE): {mse:.6f}\n")
mse_zero_noise = torch.mean(error_zero_noise**2).item()
print(f"\n Rollout Mean Squared Error (MSE) with zero noise: {mse_zero_noise:.6f}\n")




# Visualize a few frames (e.g., at index 0, 5, 10, 15 from the predicted sequence)
num_frames_to_plot = 4

fig, axes = plt.subplots(num_frames_to_plot, 3, figsize=(15, num_frames_to_plot * 4))
#fig.suptitle('Rollout Prediction vs. Ground Truth vs. Error', fontsize=16)

# Determine max value across all ground truth and predictions for consistent color mapping
vmax_val = max(predicted_sequence.max().item(), ground_truth_sequence.max().item())
vmin_val = min(predicted_sequence.min().item(), ground_truth_sequence.min().item())

# For error plot, set vmin and vmax symmetrically around 0 for a diverging colormap
max_abs_error = torch.max(torch.abs(error)).item()
max_abs_error = torch.max(torch.abs(error_zero_noise)).item()

error_vmin = -max_abs_error
error_vmax = max_abs_error

plt.rcParams['font.size'] = 14

for i in range(num_frames_to_plot):
    # Calculate index in the predicted/ground truth sequence
    # The predicted sequence has T-1 timesteps, so indices 0 to T-2.
    # Let's plot at evenly spaced indices.
    if predicted_sequence.shape[1] > 1:
        plot_idx = int(i * (predicted_sequence.shape[1] - 1) / (num_frames_to_plot - 1)) if num_frames_to_plot > 1 else 0
    else: # Only one prediction available
        plot_idx = 0

    # Predicted frame
    pred_frame = predicted_sequence[0, plot_idx, :, :, 0].cpu().numpy()
    # Ground truth frame (corresponding to the same time step)
    gt_frame = ground_truth_sequence[0, plot_idx, :, :, 0].cpu().numpy()
    # Error frame
    # err_frame = error[0, plot_idx, :, :, 0].cpu().numpy()
    err_frame = error_zero_noise[0, plot_idx, :, :, 0].cpu().numpy()

    current_time_val = full_times_sequence[0, input_len + plot_idx].item()

    # Plot Prediction
    im_pred = axes[i, 0].imshow(pred_frame, cmap='viridis', origin='lower', vmin=vmin_val, vmax=vmax_val)
    axes[i, 0].set_title(f'Prediction (Time: {current_time_val:.2f})')
    axes[i, 0].axis('off')
    fig.colorbar(im_pred, ax=axes[i, 0], fraction=0.046, pad=0.04)

    # Plot noisy ground truth
    im_gt = axes[i, 1].imshow(gt_frame, cmap='viridis', origin='lower', vmin=vmin_val, vmax=vmax_val)
    axes[i, 1].set_title(f'Ground Truth (Time: {current_time_val:.2f})')
    axes[i, 1].axis('off')
    fig.colorbar(im_gt, ax=axes[i, 1], fraction=0.046, pad=0.04)

    # Plot Error
    im_err = axes[i, 2].imshow(err_frame, cmap='RdBu_r', origin='lower', vmin=error_vmin, vmax=error_vmax) # Changed colormap and vmin/vmax
    axes[i, 2].set_title(f'Error compared to zero noise (Time: {current_time_val:.2f})')
    axes[i, 2].axis('off')
    fig.colorbar(im_err, ax=axes[i, 2], fraction=0.046, pad=0.04)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('prediction.pdf', bbox_inches='tight', pad_inches = 0)
plt.show()


from conservation import plot_mass, plot_mass_der, mass, time_der
plot_mass(ground_truth_sequence,predicted_sequence, full_times_sequence, input_len)
plot_mass_der(ground_truth_sequence,predicted_sequence, full_times_sequence, input_len)

from rollout_error import rmse
rmse(error, full_times_sequence, input_len)

from initial_time import rmse_initial_time
# rmse_initial_time(model, data_sample,times_sample, data_std, data_mean)


mass_gt = mass(ground_truth_sequence).squeeze().cpu().numpy()
mass_pred = mass(predicted_sequence).squeeze().cpu().numpy()


mass_timeder_gt = time_der(mass(ground_truth_sequence))
target_len = len(full_times_sequence[0, input_len:].cpu().numpy())
current_len = len(mass_timeder_gt)
if current_len < target_len:
    pad_width = target_len - current_len
    mass_timeder_gt = np.pad(mass_timeder_gt, (pad_width,0), constant_values=np.nan)

mass_timeder_pred = time_der(mass(predicted_sequence))
target_len = len(full_times_sequence[0, input_len:].cpu().numpy())
current_len = len(mass_timeder_pred)
if current_len < target_len:
    pad_width = target_len - current_len
    mass_timeder_pred = np.pad(mass_timeder_pred, (pad_width,0), constant_values=np.nan)



rmse_over_time = torch.sqrt(torch.mean(error**2, dim=(2, 3, 4))).squeeze().cpu().numpy()



# Save to CSV
import pandas as pd
th = pd.DataFrame({
    'Times': full_times_sequence[0, input_len:].cpu().numpy(),
    'Ground Truth Mass': mass_gt,
    'Predicted Mass':  mass_pred,
    'Ground Truth Mass Time Derivative': mass_timeder_gt,
    'Predicted Mass Time Derivative':  mass_timeder_pred,
    'RMSE': rmse_over_time
    })
th.to_csv('massoutputforthisrun.csv', index=False)

