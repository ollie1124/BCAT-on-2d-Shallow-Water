import os
import h5py
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from embedder import PatchEmbedder

import torch.nn as nn
# Import PatchEmbedder here because BCAT uses it directly.
from embedder import PatchEmbedder
# from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention as sdpa_kernel
from torch._C import _SDPBackend as SDPBackend



def block_lower_triangular_mask(block_size, block_num):
    """
    Create a block lower triangular boolean mask. (upper right part will be 1s, and represent locations to ignore.)

    """
    # Seolhwa:  block_size = number of blocks within a time frame
    #           block_num = number of time frames

    matrix_size = block_size * block_num
    lower_tri_mask = torch.tril(torch.ones(matrix_size, matrix_size, dtype=torch.bool))
    block = torch.ones(block_size, block_size, dtype=torch.bool)
    blocks = torch.block_diag(*[block for _ in range(block_num)])
    final_mask = torch.logical_or(lower_tri_mask, blocks)

    return torch.zeros_like(final_mask, dtype=torch.float32).masked_fill_(~final_mask, float("-inf"))







# -----------------------------
# Transformer model
# -----------------------------

class BCAT(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.x_num = params.x_num
        self.max_output_dim = params.data.max_output_dimension  # Seolhwa: number of PDE channels
        self.max_data_len = params.max_time_len   # Seolhwa: max_data_len = maximu sequence length
        self.embedder = PatchEmbedder(self.params, self.x_num, self.max_output_dim)
        # self.input_proj = nn.Linear(2, d_model)
        # self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        # nn.init.normal_(self.pos_emb, mean=0.0, std=0.01)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.params.dim,
            nhead=self.params.nhead,
            dim_feedforward=self.params.dim_feedforward,
            dropout=self.params.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=self.params.num_layers)
        self.seq_len_per_step = self.params.patch_num**2 # Seolhwa: image split into patch_num × patch_num

        # mask
        mask = block_lower_triangular_mask(self.seq_len_per_step, self.max_data_len) # Seolhwa: generates mask once
        self.register_buffer("mask", mask, persistent=False) # This mask is stored in the device and will not be trained.
        self.block_mask = mask
        self.mask = mask

    def forward(self, data, times, input_len: int, **kwargs):
        # The data and times are already sliced in the training loop (e.g., data[:, :-1], times[:, :-1])
        # So, removing the redundant slicing here, as it was causing data to be too short.
        # data = data[:, :-1]  # ignore last timestep for autoregressive training (b, t_num-1, x_num, x_num, data_dim)
        # times = times[:, :-1]  # (bs/1, t_num-1, 1)

        # Ensure times has the correct shape (B, T, 1) for time_proj, as it comes as (B, T) from DataLoader.
        # The original code might have implicitly expected times to already be (B, T, 1) or handled a (B, T) input differently.
        # Explicitly unsqueeze to ensure (B, T, 1) for the Linear layer in time_proj.
        if times.dim() == 2: # Check if times is (B, T) and unsqueeze if necessary.
            times = times.unsqueeze(-1)

        # Encoding
        # Ensure embedder output is flattened to (Batch, SequenceLength, EmbeddingDimension) for the Transformer
        data = self.embedder.encode(data, times, mode="flatten")  # (bs, data_len, dim) # Seolhwa: This calls "encoder" function in embedder.
        # Seolhwa: we patchfied & embedded the patch vectors to the latent space (of dimension = dim).
        """
        Step 2: Transformer
            data_input:   Tensor     (bs, data_len, dim)
        """
        data_len = data.size(1) # Seolhwa: data_len is the sequence length after patchifying, which should be (t_num-1)*patch_num*patch_num for training. For inference, it will grow as we autoregressively predict more time steps.
        mask = self.mask[:data_len, :data_len]
        # Removed the sdpa_kernel context manager to allow PyTorch to choose the best backend automatically.
        data_encoded = self.transformer(data, mask=mask)  # (bs, data_len, dim) # Seolhwa: predicts next time step autoregressively.
        # nn.TransformerEncoderLayer's forward method: (embedded) token -> token + self-attention -> output token = token + feed forward, where feed-forward is a 2-layer perceptron with activation in between. So, the output token is a function of all previous tokens (due to self-attention) and the feed-forward network.
        """
        Step 3: Decode data
        """

        # input_seq_len is meant to indicate how many initial tokens should NOT be decoded.
            # Seolhwa: input_len is seto to 1 in our case, which means input_seq_len = 0, so all tokens will be decoded. If input_len were 2, then input_seq_len would be 1, meaning the first token is not decoded (since it's an input), and decoding starts from the second token.
        # If input_len is 1 (meaning the first time step is input to predict the second),
        # then input_seq_len should be 0 to decode all tokens (predictions for the whole sequence).
        input_seq_len = (input_len - 1) * self.seq_len_per_step
        data_output = data_encoded[:, input_seq_len:]  # (bs, output_len*patch_num*patch_num, dim)
        

        # Pass the correct mode to the decoder as well, depending on what it expects.
        # Assuming the decode method also has a mode argument and expects a flattened input.
        data_output = self.embedder.decode(data_output, mode="flatten")  # (bs, output_len, x_num, x_num, data_dim)
        return data_output


    def summary(self): # Seolhwa: counts trainable parameters.
        s = "\n"
        s += f"\tEmbedder:        {sum([p.numel() for p in self.embedder.parameters() if p.requires_grad]):,}\n"
        s += f"\tTransformer:    {sum([p.numel() for p in self.transformer.parameters() if p.requires_grad]):,}"
        return s