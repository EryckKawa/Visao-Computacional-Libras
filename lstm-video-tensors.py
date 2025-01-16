import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

# Function to load tensors from a directory
def load_video_tensors(tensor_dir):
    """
    Load all .npy tensors from a directory.

    Args:
        tensor_dir (str): Path to the directory containing .npy files.

    Returns:
        list: A list of loaded tensors.
    """
    tensors = []
    filenames = sorted(os.listdir(tensor_dir))  # Sort filenames for consistent loading
    for filename in filenames:
        if filename.endswith(".npy"):
            tensor_path = os.path.join(tensor_dir, filename)
            tensors.append(np.load(tensor_path))
    return tensors

# Example usage of load_video_tensors
tensor_dir = "./VIDEO-TENSORS"  # Update this path to your tensor directory
tensors = load_video_tensors(tensor_dir)

# Preparing data for deep learning
# Combine all tensors into a single dataset
X = np.array(tensors)  # Shape: (number of videos, frames, landmarks, coordinates)

# Example: Create dummy labels (adjust based on your classification task)
num_classes = 5  # Number of classes in your dataset
num_videos = X.shape[0]
y = [0] * 18 + [1] * 18 + [2] * 18 + [3] * 18 + [4] * 18  # Dummy labels

# Stratified train-test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# One-hot encoding the labels
y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_val = tf.keras.utils.to_categorical(y_val, num_classes=num_classes)

# Reshape the input data to match the model's expected input shape
X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], -1))  # Combine landmarks and coordinates
X_val_reshaped = X_val.reshape((X_val.shape[0], X_val.shape[1], -1))

# Defining early stopping
early_stopping = EarlyStopping(
    monitor="val_loss",  # Monitora a perda de validação
    patience=10,          # Número de épocas para esperar antes de parar o treinamento (ajuste conforme necessário)
    restore_best_weights=True,  # Restaura os melhores pesos do modelo quando o treinamento é interrompido
    mode="min",          # "min" indica que a perda deve diminuir para melhorar
    verbose=1            # Exibe informações sobre o processo de early stopping
)

# Define the deep learning architecture
model = Sequential([
    LSTM(128, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]), return_sequences=True),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(num_classes, activation="softmax")
])

# Compile the model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
model.fit(
    X_train_reshaped, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val_reshaped, y_val),
    callbacks=[early_stopping]
)

# Save the trained model
model.save("video_classification_model.h5")

print("Model trained and saved successfully!")