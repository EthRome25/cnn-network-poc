import io
import base64
import json
import os
import sys
from typing import List, Dict, Any

from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-GUI backend for server environments
import matplotlib.pyplot as plt

from tensorflow import keras
import protected_data

# Configuration
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(__file__), "trained-model.keras"))
IMAGE_SIZE = (128, 128)  # must match training

# Lazy-loaded singleton model
_model = None
_labels: List[str] = []

IEXEC_OUT = os.getenv('IEXEC_OUT')

def load_model_and_labels():
    """Load the trained model and infer class labels."""
    global _model, _labels
    if _model is None:
        try:
            _model = keras.models.load_model(MODEL_PATH)
            # Try to infer labels:
            num_classes = int(_model.output_shape[-1])
            # Default labels used in the notebook/dataset (alphabetical):
            default_labels = ["glioma", "meningioma", "notumor", "pituitary"]
            if num_classes == len(default_labels):
                _labels = default_labels
            else:
                _labels = [f"class_{i}" for i in range(num_classes)]
            print(f"[INFO] Model loaded successfully. Classes: {_labels}")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            raise
    return _model, _labels

def plot_prediction(image: Image.Image, labels: List[str], probs: List[float]) -> str:
    """Create a visualization of the prediction results."""
    # Create a figure similar to the notebook: image + horizontal bar chart
    fig = plt.figure(figsize=(8, 8))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.imshow(image)
    ax1.axis("off")
    ax1.set_title("Input image")

    ax2 = fig.add_subplot(2, 1, 2)
    bars = ax2.barh(labels, probs)
    ax2.set_xlabel("Probability")
    ax2.set_xlim(0, 1)
    try:
        ax2.bar_label(bars, fmt='%.2f')
    except Exception:
        pass
    fig.tight_layout()
    
    # Save plot as base64 string
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return b64


def preprocess_image(img: Image.Image) -> np.ndarray:
    # Resize and normalize like in the notebook
    img_resized = img.resize(IMAGE_SIZE)
    arr = np.asarray(img_resized)
    # Ensure 3 channels
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[-1] == 4:
        # drop alpha
        arr = arr[..., :3]
    arr = np.expand_dims(arr, axis=0).astype("float32") / 255.0
    return arr, img_resized

def predict_brain_tumor(image_data: bytes) -> Dict[str, Any]:
    """
    Main prediction function that processes image data and returns prediction results.
    This is the core function that will be called by iExec TEE.
    """
    try:
        # Load image from bytes
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Load model
        model, labels = load_model_and_labels()
        
        # Preprocess image
        x, img_resized = preprocess_image(img)
        
        # Make prediction
        preds = model.predict(x, verbose=0)
        probs = preds[0].tolist()
        
        # Clip/normalize probabilities
        probs = [float(max(0.0, min(1.0, p))) for p in probs]
        
        # Map probabilities to labels
        if len(probs) != len(labels):
            if len(probs) < len(labels):
                labels = labels[:len(probs)]
            else:
                probs = probs[:len(labels)]
        
        prob_map: Dict[str, float] = {label: float(p) for label, p in zip(labels, probs)}
        
        # Get predicted label
        pred_idx = int(np.argmax(probs)) if probs else -1
        predicted_label = labels[pred_idx] if 0 <= pred_idx < len(labels) else None
        
        # Create visualization
        plot_b64 = plot_prediction(img_resized, labels, probs)
        
        return {
            "predicted_label": predicted_label,
            "probabilities": prob_map,
            "plot_base64_png": plot_b64,
            "status": "success"
        }
        
    except Exception as e:
        return {
            "error": f"Prediction failed: {str(e)}",
            "status": "error"
        }

def main():
    """
    Main function for iExec TEE compatibility.
    This function can be called directly by iExec with input data.
    """
    try:
        load_model_and_labels()
        try:
            # The protected data mock created for the purpose of this Hello World journey
            # contains an object with a key "secretText" which is a string
            protected_image = protected_data.getRawValue('Tr-me_0011.jpg')
        except Exception as e:
            print('It seems there is an issue with your protected data:', e)
            raise

        result = predict_brain_tumor(protected_image)
        print(result)

        # Save result as JSON
        result_json = json.dumps(result)
        with open(IEXEC_OUT + '/result.txt', 'w') as f:
            f.write(result_json)
        computed_json = {'deterministic-output-path': IEXEC_OUT + '/result.txt'}
    except Exception as e:
        print(f"Error: {e}")
        computed_json = {'deterministic-output-path': IEXEC_OUT,
                         'error-message': f'Oops something went wrong: {str(e)}'}
    finally:
        with open(IEXEC_OUT + '/computed.json', 'w') as f:
            json.dump(computed_json, f)

if __name__ == "__main__":
    main()
