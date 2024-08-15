import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

# Step 1: Collect data to train the neural network
def collect_data(num_samples):
    # Generate data using known properties of the Riemann function zeros
    data = []
    labels = []
    for _ in range(num_samples):
        real_part = 0.5  # Real part of the Riemann function zeros for the hypothetical axis
        imaginary_part = np.random.uniform(0, 30)  # Примерный диапазон
        data.append([real_part])
        labels.append([real_part, imaginary_part])
    return np.array(data), np.array(labels)

if __name__ == "__main__":
    num_samples = 1000
    epochs = 200
    batch_size = 32

    print("Collect data...")
    data, labels = collect_data(num_samples)
    input_shape = data.shape[1]