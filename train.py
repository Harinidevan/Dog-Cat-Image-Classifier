import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

# -----------------------------
# Configuration
# -----------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# Dataset Paths
train_dir = "small_dataset/train"
validation_dir = "small_dataset/validation"
test_dir = "small_dataset/test"

# -----------------------------
# Data Preprocessing
# -----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# -----------------------------
# Build MobileNetV2 Model
# -----------------------------
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# -----------------------------
# Compile Model
# -----------------------------
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Display Model Summary
model.summary()

# -----------------------------
# Train Model
# -----------------------------
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS
)

# -----------------------------
# Create models folder
# -----------------------------
os.makedirs("models", exist_ok=True)

# Save Model
model.save("models/dog_cat_classifier.keras")

print("\n✅ Model saved successfully!")

# -----------------------------
# Evaluate Model
# -----------------------------
loss, accuracy = model.evaluate(test_generator)

print(f"\nTest Accuracy : {accuracy * 100:.2f}%")
print(f"Test Loss     : {loss:.4f}")