import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import build_model
import os
from tensorflow.keras.applications.resnet50 import preprocess_input

data_dir = 'data'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

# Image parameters
img_height, img_width = 224, 224
batch_size = 32

# Data augmentation for training
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Load datasets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

# Build model
model = build_model(input_shape=(img_height, img_width, 3))

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model (capped for faster CPU training)
epochs = 10
history = model.fit(
    train_generator,
    steps_per_epoch=min(200, train_generator.samples // batch_size),
    validation_data=val_generator,
    validation_steps=min(50, val_generator.samples // batch_size),
    epochs=epochs
)

# Save model
os.makedirs('models', exist_ok=True)
model.save('models/deepfake_detector.h5')

print("Training completed and model saved.")
