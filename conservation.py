import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import DataLoader # Needed for val_loader re-creation if not already imported

# Conserved quantities and their derivatives

def mass(sequence):
  # sequence is (B, T, x_num, x_num, c)
  return torch.sum(sequence, dim=(2, 3))

# Can define more conserved quantities.

def time_der(mass):
  # mass is (B, T, x_num, x_num, c)
  return (mass[:, 1:] - mass[:, :-1]).squeeze().cpu().numpy()


def plot_mass(ground_truth_sequence,predicted_sequence, full_times_sequence, input_len):
    # Calculate time derivatives for both ground truth and predicted sequences
    plot_values_gt = mass(ground_truth_sequence).squeeze().cpu().numpy()
    plot_values_pred = mass(predicted_sequence).squeeze().cpu().numpy()

    # Ensure times_for_plot matches the length of the sequences
    # The sequences `ground_truth_sequence` and `predicted_sequence` correspond to times
    # starting from `full_times_sequence[0, input_len:]` within the `full_times_sequence`.

    times_for_plot = full_times_sequence[0, input_len:].cpu().numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(times_for_plot, plot_values_gt, marker='o', linestyle='-', label='Ground Truth Mass')
    plt.plot(times_for_plot, plot_values_pred, marker='x', linestyle='--', label='Predicted Mass')
    plt.title('Mass: Ground Truth vs. Prediction Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Mass')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_mass_der(ground_truth_sequence,predicted_sequence, full_times_sequence, input_len):
    # Calculate time derivatives for both ground truth and predicted sequences
    plot_values_gt = time_der(mass(ground_truth_sequence))
    plot_values_pred = time_der(mass(predicted_sequence))

    times_for_plot = full_times_sequence[0, input_len + 1 :].cpu().numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(times_for_plot, plot_values_gt, marker='o', linestyle='-', label='Ground Truth Mass Derivative')
    plt.plot(times_for_plot, plot_values_pred, marker='x', linestyle='--', label='Predicted Mass Derivative')
    plt.title('Time Derivative of Mass: Ground Truth vs. Prediction')
    plt.xlabel('Time (s)')
    plt.ylabel('Mass Derivative')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()