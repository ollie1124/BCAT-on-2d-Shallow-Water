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
            t_step=1
        ),

        # We can define other systems too.
          # e.g. diff_reaction=types.SimpleNamespace(...),

        # Below are default values:
        x_num=128,# This is for shallow water dataset, but for other dataset, this number might change.
        t_num=101,# Other systems (e.g. NS) have different time steps... (Reduced from 101 for performance)
        random_start={'train': False, 'val': False, 'test': False, 'start_max': 0},
        train_val_test_ratio=[0.8, 0.1, 0.1],
        tie_fields=True,
        max_output_dimension=1
    ),
    overfit_test=False,
    noise=0,
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
    batch_size=20, # Reduced to 2 as per user's current setting

    # Latent space encoding/decoding params
    dim = 64, # originally 1024, embedding dimension, a.k.a. hidden dimension of attention, latent space dimension, etc. (Reduced from 256 for performance)
    x_num = 128,
    patch_size = 8,
    patch_num = 128//8, # params.x_num// params.patch_size,
    patch_num_output = 128//8, #params.patch_num, # in case the decoder (de)compresses the result

    # Transformer hyperparameters
    # These are in Table 9 of the BCAT paper.  I downsized the parameters.
    nhead =2, # 8, originally. (Reduced from 4 for performance)
    num_layers=2,# 12, originally (Reduced from 8 for performance)
    dim_feedforward=64, # 275, originally (Reduced from 128 for performance)
    max_len=128,
    max_time_len =20, # Must match params.data.t_num (Reduced from 101 for performance)
    dropout=0

)

# Create a dummy symbol_env object
symbol_env = types.SimpleNamespace()


print(f"data_path has been set to: {params.data.shallow_water.data_path}")




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
# Training model
# ==============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BCAT(params).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4,
)

criterion = torch.nn.MSELoss()

# == Training loop (simplified) ==============================
batch = next(iter(train_loader))


num_epochs = 100

for epoch in range(num_epochs):

    model.train()

    total_loss = 0.0

    for batch in train_loader:

        # -------------------------------------------------
        # batch data
        # -------------------------------------------------

        data = batch["data"].to(device)     # (B,T,H,W,C)
        times = batch["t"].to(device)       # (B,T)

        # -------------------------------------------------
        # autoregressive shift
        # -------------------------------------------------

        input_data = data[:, :-1]
        target_data = data[:, 1:]

        input_times = times[:, :-1] # Shape (B, T-1)
        # Ensure input_times has a channel dimension for the time projection layer (B, T-1, 1)
        input_times = input_times.unsqueeze(-1)

        # The original BCAT fwd method's comment suggests input_len should be 1 for training
        # to ensure the full predicted sequence is decoded. With input_len = input_times.size(1)
        # (which is 100), only the last timestep's prediction would be extracted.
        input_len_for_fwd = 1

        # -------------------------------------------------
        # forward pass
        # -------------------------------------------------

        pred = model(input_data, input_times, input_len_for_fwd)

        # expected shape:
        # pred: (B,T-1,H,W,C)

        loss = criterion(pred, target_data)

        # -------------------------------------------------
        # backward
        # -------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.6f}")

    # Validation loop

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for batch in val_loader:

            data = batch["data"].to(device)
            times = batch["t"].to(device)

            input_data = data[:, :-1]
            target_data = data[:, 1:]

            input_times = times[:, :-1]

            pred = model(input_data, input_times, input_len_for_fwd)

            loss = criterion(pred, target_data)

            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch:03d} | Val Loss: {avg_val_loss:.6f}")

    # Saving check point
    checkpoint_path = "bcat_shallow_water.pt"

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )





# ================================================
# --- Code to execute rollout and plot results ---
# ================================================
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import DataLoader # Needed for val_loader re-creation if not already imported

# Load the trained model checkpoint
checkpoint_path = "bcat_shallow_water.pt"
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval() # Set model to evaluation mode

print("Model loaded and set to evaluation mode.")

# Get one batch from the validation loader for rollout
# Re-create val_loader to ensure it's fresh if it was exhausted
val_loader = DataLoader(
    val_dataset,
    batch_size=params.batch_size,
    num_workers=params.num_workers_eval,
    shuffle=False # For consistent validation results
)

# Get a single batch for testing rollout
val_batch = next(iter(val_loader))

# Prepare inputs for rollout_model
# We take the first sample from the batch
data_sample = val_batch["data"][0:1, :, :, :, :].to(device) # (1, T, H, W, C)
times_sample = val_batch["t"][0:1, :].to(device) # (1, T)

# The input to rollout_model should be the first timestep of the data_sample
initial_input = data_sample[:, :1, :, :, :]

# The full time sequence for the prediction horizon
full_times_sequence = times_sample

# input_len is 1 because we are providing only the first frame as input
input_len = 1

# data_mask: a mask of ones if no specific channels are to be masked out
data_mask = torch.ones_like(initial_input[:, 0:1, :, :, :]).to(device)

print(f"Initial input shape for rollout: {initial_input.shape}")
print(f"Full times sequence shape for rollout: {full_times_sequence.shape}")

# Perform rollout prediction
with torch.no_grad():
    predicted_sequence = rollout_model(model, initial_input, full_times_sequence, input_len, data_mask)

print(f"Predicted sequence shape: {predicted_sequence.shape}")

# Compare with ground truth
ground_truth_sequence = data_sample[:, input_len:, :, :, :]

# Visualize a few frames (e.g., at index 0, 5, 10, 15 from the predicted sequence)
num_frames_to_plot = 4

fig, axes = plt.subplots(num_frames_to_plot, 2, figsize=(10, num_frames_to_plot * 5))
fig.suptitle('Rollout Prediction vs. Ground Truth', fontsize=16)

# Determine max value across all ground truth and predictions for consistent color mapping
vmax_val = max(predicted_sequence.max().item(), ground_truth_sequence.max().item())
vmin_val = min(predicted_sequence.min().item(), ground_truth_sequence.min().item())

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

    current_time_val = full_times_sequence[0, input_len + plot_idx].item()

    # Plot Prediction
    im_pred = axes[i, 0].imshow(pred_frame, cmap='viridis', origin='lower', vmin=vmin_val, vmax=vmax_val)
    axes[i, 0].set_title(f'Prediction (Time: {current_time_val:.2f})')
    axes[i, 0].axis('off')
    fig.colorbar(im_pred, ax=axes[i, 0], fraction=0.046, pad=0.04)

    # Plot Ground Truth
    im_gt = axes[i, 1].imshow(gt_frame, cmap='viridis', origin='lower', vmin=vmin_val, vmax=vmax_val)
    axes[i, 1].set_title(f'Ground Truth (Time: {current_time_val:.2f})')
    axes[i, 1].axis('off')
    fig.colorbar(im_gt, ax=axes[i, 1], fraction=0.046, pad=0.04)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
