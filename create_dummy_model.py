# create_dummy_model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# Create a very small dummy CNN model
model = Sequential([
    Conv2D(8, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # binary output
])

# Compile dummy model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Ensure 'model' folder exists
os.makedirs('model', exist_ok=True)

# Save dummy model without training
model.save('model/skin_cancer_model.h5')
print("✅ Dummy model created at model/skin_cancer_model.h5")