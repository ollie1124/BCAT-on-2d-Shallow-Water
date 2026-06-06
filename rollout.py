import torch
import numpy as np
from torch.utils.data import DataLoader # Needed for val_loader re-creation if not already imported


# Define the rollout_model function as a standalone function, correcting the 'self' argument
@torch.compiler.disable()
def rollout_model(model, data_input, times, input_len: int, data_mask, data_mean, data_std, carry_over_c=-1):
    """
        Inputs:
            model:         The trained BCAT model instance.
            data_input:    Tensor     (bs, input_len, x_num, x_num, data_dim)
            times:         Tensor     (bs/1, input_len+output_len, 1)
            data_mask:     Tensor     (1, 1, 1, 1, data_dim)
            data_mean:     Scalar     Mean used for normalization
            data_std:      Scalar     Std used for normalization
            carry_over_c:  int        Indicate channel that should be carried over,
                                        not masked out or from output (e.g. boundary mask channel)

        Output:
            data_output:     Tensor     (bs, output_len, x_num, x_num, data_dim) (denormalized)
        """
    # Ensure model is in evaluation mode
    model.eval()

    t_num = times.size(1)
    output_len = t_num - input_len
    bs, _, x_num, _, data_dim = data_input.size()

    # Initialize data_all, storing normalized data
    data_all = torch.zeros(bs, t_num, x_num, x_num, data_dim, dtype=data_input.dtype, device=data_input.device)
    data_all[:, :input_len] = (data_input - data_mean) / data_std # Normalize initial input
    cur_len = input_len

    for i in range(output_len):
        cur_data_input = data_all[:, :cur_len]  # (bs, cur_len, x_num, x_num, data_dim)

        # Prepare times for embedder: ensure shape is (bs, cur_len, 1)
        times_for_embedder = times[:, :cur_len]
        if times_for_embedder.dim() == 2:
            times_for_embedder = times_for_embedder.unsqueeze(-1)

        # Encode the current input, ensuring mode='flatten' is used
        cur_data_encoded_input = model.embedder.encode(
            cur_data_input, times_for_embedder, mode="flatten"
        )  # (bs, data_len, dim)

        # Prepare mask for the transformer
        data_len = cur_len * model.seq_len_per_step
        mask = model.mask[:data_len, :data_len]

        # Pass through the transformer
        cur_data_encoded = model.transformer(cur_data_encoded_input, mask=mask)

        # Get the new output from the last predicted patch
        new_output_encoded = cur_data_encoded[:, -model.seq_len_per_step :]

        # Decode the new output, ensuring mode='flatten' is used
        new_output = model.embedder.decode(new_output_encoded, mode="flatten")  # (bs, 1, x_num, x_num, data_dim)

        new_output = new_output * data_mask  # (bs, 1, x_num, x_num, data_dim)

        if carry_over_c >= 0:
            new_output[:, 0, :, :, carry_over_c] = data_all[:, 0, :, :, carry_over_c]

        data_all[:, cur_len : cur_len + 1] = new_output
        cur_len += 1

    # Denormalize the entire predicted sequence before returning
    predicted_normalized_sequence = data_all[:, input_len:]
    predicted_denormalized_sequence = predicted_normalized_sequence * data_std + data_mean
    return predicted_denormalized_sequence
