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

# Шаг 2: Создаем архитектуру нейронной сети
def create_neural_network(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(2)  # Выходные данные: реальные и мнимые части
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

# Шаг 3: Обучаем нейронную сеть
def train_neural_network(model, data, labels, epochs=100, batch_size=16):
    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=1)
    return history

# Шаг 4: Визуализируем результаты
def plot_training_history(history):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Training and Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.show()

# Шаг 5: Оцениваем точность модели
def evaluate_model(model, data, labels):
    predictions = model.predict(data)
    mse = mean_squared_error(labels, predictions)
    print(f'Mean Squared Error: {mse}')
    return predictions

# Основная часть программы
if __name__ == "__main__":
    num_samples = 1000
    epochs = 200
    batch_size = 32

    print("Collect data...")
    data, labels = collect_data(num_samples)
    input_shape = data.shape[1]

    print("Создание модели...")
    model = create_neural_network(input_shape)

    print("Обучение модели...")
    history = train_neural_network(model, data, labels, epochs, batch_size)

    print("Визуализация результатов...")
    plot_training_history(history)

    print("Оценка модели...")
    predictions = evaluate_model(model, data, labels)

    # Дополнительная визуализация сравнения прогнозов с реальными значениями
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(labels[:, 0], labels[:, 1], label='Real')
    ax.scatter(predictions[:, 0], predictions[:, 1], label='Predictions', marker='^')
    ax.set_xlabel('Real Part')
    ax.set_ylabel('Imaginary Part')
    ax.set_zlabel('Value')
    ax.legend()
    plt.show()