import sys
print(sys.executable)

import torch
from torch.utils.data import DataLoader
from all_datasets import ShallowWater2D


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
            t_step=4
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
    noise=0.1,
    noise_type = "additive", # "multiplicative",
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
    batch_size=2, # Reduced to 2 as per user's current setting

    # Latent space encoding/decoding params
    dim = 128, # originally 1024, embedding dimension, a.k.a. hidden dimension of attention, latent space dimension, etc. (Increased from 64)
    x_num = 128,
    patch_size = 8,# originally 128 (64 for fine tuning)
    patch_num = 128//8, # params.x_num// params.patch_size,
    patch_num_output = 128//8, #params.patch_num, # in case the decoder (de)compresses the result

    # Transformer hyperparameters
    # These are in Table 9 of the BCAT paper.  I downsized the parameters.
    nhead = 4, # 8, originally. (Increased from 2)
    num_layers=4,# 12, originally (Increased from 2)
    dim_feedforward=128, # 275, originally (Increased from 64)
    max_len=128,
    max_time_len =100, # Must match params.data.t_num (Reduced from 101 for performance)
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
    shuffle=False, # Shuffle handled within the dataset for training
    num_workers=params.num_workers
)

# Assuming a total of 1000 samples in the dataset for range calculation.
# The `get_iter_range` method calculates the number of samples for the 'train' split
# based on `train_val_test_ratio` and `num_workers`.
# It's important that this value reflects the actual number of files in the HDF5.
# A more robust check would involve opening the HDF5 file and getting its length.

# To get the actual number of items in the HDF5 file for printing:
import h5py
print(f'\n{params.data.shallow_water.data_path}')


with h5py.File(params.data.shallow_water.data_path, 'r') as hf:
    total_samples_in_file = len(hf)

print(f"Successfully initialized ShallowWater2D dataset.")
print(f"Number of samples accessible to this worker (train split): {len(shallow_water_dataset.get_iter_range(total_samples_in_file))}")

# Example of iterating through the data (showing first batch)
print("\nFetching first batch from DataLoader...")
for i, sample in enumerate(dataset_loader):
    print(f"Batch {i+1} data shape: {sample['data'].shape}")
    print(f"Batch {i+1} data dtype: {sample['data'].dtype}")
    # Display first few values from the data tensor (time was causing KeyError)
    print(f"First few data values: {sample['data'][0,:,10,10,0]}")

    print(f"\nSample {i}")
    print("Keys:", sample.keys())

    data = sample["data"]
    print("data shape:", data.shape)
    print("data dtype:", data.dtype)
    print("data min:", data.min().item())
    print("data max:", data.max().item())
    print("data mean:", data.mean().item())

    # if "t" in sample:
    #     print("t shape:", sample["t"].shape)
    #     print("t:", sample["t"])
    #     print("data",data)
    if i == 0:
        break


#==============================
# Test the samples
#==============================

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data

# Get one sample from the DataLoader
# Ensure the DataLoader is iterable, if it has already been exhausted, recreate it.
dataset_loader = data.DataLoader(
    shallow_water_dataset,
    batch_size= 2, #params.batch_size, # Now params.batch_size will be used
    shuffle=True,
    num_workers=params.num_workers
)

for i, sample in enumerate(dataset_loader):
    # We'll just take the first sample in the batch for visualization, e.g., index 0
    batch_idx_to_visualize = 1 # You can change this to any valid index within your batch_size

    data_sample = sample['data'][batch_idx_to_visualize]
    time_values = sample['t'][batch_idx_to_visualize]

    print(f"Shape of data_sample: {data_sample.shape} (timesteps, x_dim, y_dim, channels)")
    print(f"Shape of time_values: {time_values.shape}")

    # Choose a specific time step to visualize (e.g., the first one)
    # The data_sample is (timesteps, x_dim, y_dim, channels)
    for j in range(4):
      time_step_idx = 0 + 5*j
      data_2d = data_sample[time_step_idx, :, :, 0] # Assuming single channel data
      current_time = time_values[time_step_idx].item()

      print(f"Visualizing data at time index {time_step_idx} (actual time: {current_time:.4f})")
      print(f"Shape of 2D data for visualization: {data_2d.shape}")
      print(f"Min value: {data_2d.min().item():.4f}, Max value: {data_2d.max().item():.4f}")

      plt.figure(figsize=(8, 6))
      plt.imshow(data_2d.cpu().numpy(), cmap='viridis', origin='lower') # 'origin' for correct orientation
      plt.colorbar(label='Value')
      plt.title(f'2D Shallow Water Data at Time: {current_time:.4f}')
      plt.xlabel('X-dimension')
      plt.ylabel('Y-dimension')
      plt.show()

    # Break after visualizing the first batch's first sample
    break