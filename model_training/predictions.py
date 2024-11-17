import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from typing import Union

# Load the models when importing "predictions.py"
try:
    model_elbow_frac = tf.keras.models.load_model("weights/ResNet50_Elbow_frac.h5")
    model_hand_frac = tf.keras.models.load_model("weights/ResNet50_Hand_frac.h5")
    model_shoulder_frac = tf.keras.models.load_model("weights/ResNet50_Shoulder_frac.h5")
    model_parts = tf.keras.models.load_model("weights/ResNet50_BodyParts.h5")
except Exception as e:
    raise RuntimeError(f"Error loading models: {e}")

# Categories for each result by index
categories_parts = ["Elbow", "Hand", "Shoulder"]
categories_fracture = ['fractured', 'normal']

# Dictionary for model selection
model_dict = {
    'Parts': model_parts,
    'Elbow': model_elbow_frac,
    'Hand': model_hand_frac,
    'Shoulder': model_shoulder_frac
}

def predict(img: Union[str, np.ndarray], model: str = "Parts", verbose: int = 0) -> str:
    size = 224
    chosen_model = model_dict.get(model)
    
    if chosen_model is None:
        raise ValueError(f"Model '{model}' is not recognized. Valid options are: {list(model_dict.keys())}")

    # Load and preprocess image
    if isinstance(img, str):
        temp_img = image.load_img(img, target_size=(size, size))
        x = image.img_to_array(temp_img)
    elif isinstance(img, np.ndarray):
        if img.shape[:2] != (size, size):
            raise ValueError(f"Input image must be of size ({size}, {size})")
        x = img
    else:
        raise TypeError("Input image must be a file path or a numpy array")

    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    
    try:
        prediction = np.argmax(chosen_model.predict(images, verbose=verbose), axis=1)
    except Exception as e:
        raise RuntimeError(f"Error during prediction: {e}")

    # Choose the category and get the string prediction
    if model == 'Parts':
        prediction_str = categories_parts[prediction.item()]
    else:
        prediction_str = categories_fracture[prediction.item()]

    return prediction_str