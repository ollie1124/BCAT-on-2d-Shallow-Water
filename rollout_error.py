import matplotlib.pyplot as plt
import torch
import numpy as np

def rmse(error, full_times_sequence, input_len):
    # error is (B, T, x_num, x_num, c)
    mse_over_time = torch.sqrt(torch.mean(error**2, dim=(2, 3, 4))).squeeze().cpu().numpy()

    times_for_mse_plot = full_times_sequence[0, input_len:].cpu().numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(times_for_mse_plot, mse_over_time, marker='o', linestyle='-', color='red', label='RMSE Over Time')
    plt.title('Root of Mean Squared Error (MSE) vs. Time for First Sample')
    plt.xlabel('Time (s)')
    plt.ylabel('Root Mean Squared Error')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()