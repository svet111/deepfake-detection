import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

def build_model(input_shape=(224, 224, 3)):
    """
    Build a deepfake detection model using ResNet50 as base.
    """
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)  # Binary classification
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze base layers for transfer learning
    for layer in base_model.layers:
        layer.trainable = False
    
    return model
