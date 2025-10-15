# train_model.py
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# --- Set Your Dataset Path ---
dataset_path = r'C:\Users\niles\OneDrive\Desktop\Minor_Project\dataset'

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

print(f"✅ Using dataset from: {dataset_path}")

# Image settings
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 10

# Data Generators
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',  # Binary classification: healthy vs rust
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# Save class indices
class_names = list(train_generator.class_indices.keys())
print("🎯 Classes:", class_names)  # Should be ['healthy', 'rust']

# Build Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary output
])

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train
print("🚀 Training started...")
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    verbose=1
)

# Save model and class names
os.makedirs('model', exist_ok=True)
model.save('model/wheat_disease_model.h5')
with open('model/class_names.txt', 'w') as f:
    f.write('\n'.join(class_names))

print("✅ Model saved to 'model/wheat_disease_model.h5'")
print("✅ Class names saved!")