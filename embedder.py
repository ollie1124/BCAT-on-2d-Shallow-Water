# ---------------------
# Simplified version
# ----------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
"""
The einops library allows reshaping tensors more transparent.
e.g. rearrange(x, "b t h w c -> (b t) c h w")
"""
from einops import rearrange
# from einops.layers.torch import Rearrange # Seolhwa: This is not used here so can be removed.


# Seolhwa: The following function re-initializes the given layer to have small outputs, which is not used in our case.
# def layer_initialize(layer, mode="zero", gamma=0.01):
#     # re-initialize given layer to have small outputs
#     if mode == "zero":
#         nn.init.zeros_(layer.weight) # Seolhwa: makes layer initially output nearly zero.
#         if layer.bias is not None:
#             nn.init.zeros_(layer.bias)
#     elif mode == "uniform": # Seolhwa: small random initialization.
#         nn.init.uniform_(layer.weight, -gamma, gamma)
#         if layer.bias is not None:
#             nn.init.uniform_(layer.bias, -gamma, gamma)
#     else:
#         raise ValueError(f"Unknown mode {mode}")



class PatchEmbedder(nn.Module):
    """
    Preprocess data (break into patches) and embed them into target dimension.
    """

    def __init__(self, params, x_num, max_output_dim):
        super().__init__()
        self.config = params

        self.dim = self.config.dim # embedding dimension
        self.data_dim = max_output_dim # Seolhwa: number of channels
        # self.data_dim = self.config.data.max_output_dimension # Seolhwa: number of channels
        act = nn.GELU # get_activation("gelu")

        assert (
            x_num % self.config.patch_num == 0
        ), f"x_num must be divisible by patch_num, x_num: {x_num}, patch_num: {config.patch_num}"
        self.patch_resolution = x_num // self.config.patch_num  # (self.patch_resolution)^2 = number of pixels within a patch
        self.patch_dim = self.data_dim * self.patch_resolution * self.patch_resolution  # dimension per patch

        assert (
            x_num % self.config.patch_num_output == 0
        ), f"x_num must be divisible by patch_num_output, x_num: {x_num}, patch_num_output: {config.patch_num_output}"
        self.patch_resolution_output = (
            x_num // self.config.patch_num_output
        )  # resolution of one space dimension for each patch in output
        self.patch_dim_output = (
            self.data_dim * self.patch_resolution_output * self.patch_resolution_output
        )  # dimension per patch in output

        ## for encoder part
        # Seolhwa: Define the neural networks that will be used later in encoding/decoding.

        self.patch_position_embeddings = nn.Parameter(torch.randn((1, 1, self.config.patch_num, self.config.patch_num, self.dim))) # get_embeddings((1, 1, config.patch_num, config.patch_num, self.dim))

        # self.time_embed_type = config.get("time_embed", "continuous") # commented out because params don't have "get".
        self.time_embed_type = getattr(self.config, "time_embed", "continuous")
        match self.time_embed_type:
            case "continuous":
                self.time_proj = nn.Sequential(
                    nn.Linear(1, self.dim),
                    act(),
                    nn.Linear(self.dim, self.dim),
                )
            case "learnable":
                self.time_embeddings = nn.Parameter(torch.randn((1, getattr(self.config, "max_time_len", 20), 1, 1, self.dim))) # get_embeddings((1, config.get("max_time_len", 20), 1, 1, self.dim))

        # Seolhwa: The following is the actual patch extraction.
        # regular vit patch embedding
        self.in_proj = nn.Conv2d(
            in_channels=self.data_dim,
            out_channels=self.dim,
            kernel_size=self.patch_resolution,# Seolhwa: To ensure non-overlapping patches
            stride=self.patch_resolution,
        )
        # Seolhwa: the following is a small nonlinear refinement after patch embedding.
        self.conv_proj = nn.Sequential(
            act(),
            nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=1, stride=1),
        )

        ## for decoder part

        self.conv_dim = getattr(self.config,"conv_dim", self.dim // 4)

        self.post_proj = nn.Sequential(
            # Rearrange("b (t h w) d -> (b t) d h w", h=self.config.patch_num_output, w=self.config.patch_num_output),
            nn.ConvTranspose2d(
                in_channels=self.dim,
                out_channels=self.conv_dim,
                kernel_size=self.patch_resolution_output,
                stride=self.patch_resolution_output,
            ),# Seolhwa: upsampling from patch space back to pixel space.
            act(),
            nn.Conv2d(in_channels=self.conv_dim, out_channels=self.conv_dim, kernel_size=1, stride=1),
            act(),
        )
        self.head = nn.Conv2d(in_channels=self.conv_dim, out_channels=self.data_dim, kernel_size=1, stride=1)

    def encode(self, data, times, mode="none"):# Seolhwa: forward pass for encoding
        """
        Input:
            data:           Tensor (bs, input_len, x_num, x_num, data_dim)
            times:          Tensor (bs, input_len, 1)
        Output:
            data:   embedded data + time embeddings + patch position embeddings
                mode:   flatten -> Tensor (bs, input_len*patch_num*patch_num, dim)
                        st      -> Tensor (bs, input_len, patch_num*patch_num, dim)
                        none    -> Tensor (bs, input_len, patch_num, patch_num, dim)
        """

        bs = data.size(0)
        data = rearrange(data, "b t h w c -> (b t) c h w")
        data = self.in_proj(data)
        data = self.conv_proj(data)  # (bs*input_len, d, patch_num, patch_num)
        data = rearrange(data, "(b t) d h w -> b t h w d", b=bs)  # (bs, input_len, p, p, dim)

        match self.time_embed_type:
            case "continuous":
                time_embeddings = self.time_proj(times)[:, :, None, None]  # (bs, input_len, 1, 1, dim)
                data = data + time_embeddings
            case "learnable":
                time_embeddings = self.time_embeddings[:, : times.size(1)]  # (bs, input_len, 1, 1, dim)
                data = data + time_embeddings

        data = data + self.patch_position_embeddings  # (b, input_len, p*p, d)

        match mode:
            case "flatten":
                return data.reshape(bs, -1, self.dim)
            case "st":
                # space time
                return rearrange(data, "b t h w c -> b t (h w) c")
            case _:
                return data

    def decode(self, data_output, mode="none"):
        """
        Input:
            data_output:
                mode:   flatten -> Tensor (bs, output_len*patch_num*patch_num, dim)
                        st      -> Tensor (bs, output_len, patch_num*patch_num, dim)
                        none    -> Tensor (bs, output_len, patch_num, patch_num, dim)
        Output:
            data_output:     Tensor (bs, output_len, x_num, x_num, data_dim)
        """
        bs = data_output.size(0)

        match mode:
            case "flatten":
                data_output = rearrange(
                    data_output,
                    "b (t h w) d -> (b t) d h w",
                    h=self.config.patch_num_output,
                    w=self.config.patch_num_output,
                )
            case "st":
                data_output = rearrange(
                    data_output,
                    "b t (h w) d -> (b t) d h w",
                    h=self.config.patch_num_output,
                    w=self.config.patch_num_output,
                )
            case _:
                data_output = rearrange(data_output, "b t h w d -> (b t) d h w")

        data_output = self.post_proj(data_output)  # (bs*output_len, data_dim, x_num, x_num)
        data_output = self.head(data_output)
        data_output = rearrange(data_output, "(b t) c h w -> b t h w c", b=bs)
        return data_output