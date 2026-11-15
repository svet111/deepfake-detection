import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os
from tensorflow.keras.applications.resnet50 import preprocess_input

data_dir = 'data'
test_dir = os.path.join(data_dir, 'test')

# Image parameters
img_height, img_width = 224, 224
batch_size = 32

# Load test data
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Load model
model = tf.keras.models.load_model('models/deepfake_detector.h5')

# Evaluate
loss, accuracy = model.evaluate(test_generator)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')

# Predictions
predictions = model.predict(test_generator)
predicted_classes = (predictions > 0.5).astype(int).flatten()

# True labels
true_classes = test_generator.classes

# Classification report
print(classification_report(true_classes, predicted_classes, target_names=['Fake', 'Real']))

# Confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)
print('Confusion Matrix:')
print(cm)
