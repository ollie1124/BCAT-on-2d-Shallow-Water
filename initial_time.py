import matplotlib.pyplot as plt
import torch
import numpy as np
from rollout import rollout_model

def rmse_initial_time(model, data_sample,times_sample, data_std, data_mean):
    input_len = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    initial_times = np.array([0,5,10,15])

    all_rmse_data = []
    all_times_data = []

    for initial_time in initial_times:
        initial_input = data_sample[:, initial_time:initial_time+1, :, :, :]
        full_times_sequence = times_sample[:,initial_time:]
        data_mask = torch.ones_like(initial_input[:, 0:1, :, :, :]).to(device)


        # Perform rollout prediction
        with torch.no_grad():
            predicted_sequence = rollout_model(model, initial_input, full_times_sequence, input_len, data_mask, data_mean, data_std)

        print(f"Predicted sequence shape for initial_time {initial_time}: {predicted_sequence.shape}")

        # Compare with ground truth
        ground_truth_sequence = data_sample[:, initial_time + input_len:, :, :, :]
        error = predicted_sequence - ground_truth_sequence

        # Calculate RMSE over time
        se_over_time = torch.sqrt(torch.mean(error**2, dim=(2, 3, 4))).squeeze().cpu().numpy()
        times_for_mse_plot = full_times_sequence[0, input_len:].cpu().numpy()

        all_rmse_data.append(se_over_time)
        all_times_data.append(times_for_mse_plot)

    print("\nFinished all rollouts for different initial times.")


    # Plot RMSE over time for different initial times
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 7))

    for i, initial_time in enumerate(initial_times):
        plt.plot(all_times_data[i], all_rmse_data[i], marker='o', linestyle='-', label=f'Initial Time: {initial_time}')

    plt.title('RMSE Over Time for Different Initial Prediction Times')
    plt.xlabel('Time (s)')
    plt.ylabel('Root Mean Squared Error')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()