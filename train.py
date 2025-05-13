
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import AdamW
import os

# Load dataset
data = pd.read_csv(
    r"E:\Rewad\Final\DNN-Deploy\data\heart_disease_risk_dataset_earlymed.csv")

# Features and labels
X = data.drop(['Heart_Risk'], axis=1)
y = data['Heart_Risk']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=44, shuffle=True
)

# Define model
model = keras.models.Sequential([
    keras.layers.Dense(8, activation='tanh', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(128, activation='sigmoid'),
    keras.layers.Dense(64, activation='tanh'),
    keras.layers.Dense(32, activation='tanh'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile model
MyOptimizer = AdamW(
    learning_rate=0.001,
    weight_decay=0.004,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    clipnorm=None,
    clipvalue=None,
    global_clipnorm=None,
    use_ema=False,
    ema_momentum=0.99,
    ema_overwrite_frequency=None,
    name="AdamW"
)
model.compile(optimizer=MyOptimizer, loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, validation_split=0.2, epochs=20, batch_size=32)

# Save model
os.makedirs("model", exist_ok=True)
model.save("model/model.keras")

print("✅ Model training complete. Saved to model/model.h5")
