import numpy as np
import matplotlib.pyplot as plt

# Load the npz file
data = np.load('/ceph/home/student.aau.dk/wb68dm/Isaac-GR00T_RL/reward_model_inference_results.npz')

# See what arrays are stored in the file
print("Arrays in the file:", data.files)

# Extract the arrays
predictions = data['preds']
targets = data['targets']

# Basic statistics
print(f"Predictions shape: {predictions.shape}")
print(f"Targets shape: {targets.shape}")
print(f"Predictions mean: {predictions.mean():.4f}, min: {predictions.min():.4f}, max: {predictions.max():.4f}")
print(f"Targets mean: {targets.mean():.4f}, min: {targets.min():.4f}, max: {targets.max():.4f}")

# Calculate metrics
mse = np.mean((predictions - targets) ** 2)
mae = np.mean(np.abs(predictions - targets))
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")

# Plot the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(range(len(predictions)), predictions, label='Predictions', alpha=0.7)
plt.scatter(range(len(targets)), targets, label='Targets', alpha=0.7)
plt.legend()
plt.title('Predictions vs Targets')
plt.xlabel('Sample Index')
plt.ylabel('Value')

plt.subplot(1, 2, 2)
plt.hist(predictions, bins=10, alpha=0.5, label='Predictions')
plt.hist(targets, bins=2, alpha=0.5, label='Targets')
plt.legend()
plt.title('Distribution of Values')

plt.tight_layout()
plt.savefig('reward_model_analysis.png')
plt.close()

print("Analysis complete! Saved plot to reward_model_analysis.png")