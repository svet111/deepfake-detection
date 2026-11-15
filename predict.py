import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load the trained model
model = tf.keras.models.load_model('models/deepfake_detector.h5')

def predict_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)[0][0]
    threshold = 0.6
    label = 'Real' if prediction > threshold else 'Fake'
    confidence = prediction if prediction > threshold else 1 - prediction
    print(f"Prediction: {label} ({confidence:.2%} confidence)")

if __name__ == '__main__':
    image_path = sys.argv[1] if len(sys.argv) > 1 else 'test_image.jpg'
    predict_image(image_path)