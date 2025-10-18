import io
import base64
import os
from typing import List, Dict, Any

from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-GUI backend for server environments
import matplotlib.pyplot as plt

from tensorflow import keras

from training_service import TrainParams, train_model_service

# Configuration
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(__file__), "trained-model.keras"))
IMAGE_SIZE = (128, 128)  # must match training

app = Flask(__name__)

# Lazy-loaded singleton model
_model = None
_labels: List[str] = []


def load_model_and_labels():
    global _model, _labels
    if _model is None:
        _model = keras.models.load_model(MODEL_PATH)
        # Try to infer labels:
        num_classes = int(_model.output_shape[-1])
        # Default labels used in the notebook/dataset (alphabetical):
        default_labels = ["glioma", "meningioma", "notumor", "pituitary"]
        if num_classes == len(default_labels):
            _labels = default_labels
        else:
            _labels = [f"class_{i}" for i in range(num_classes)]
    return _model, _labels


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


def plot_prediction(image: Image.Image, labels: List[str], probs: List[float]) -> str:
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
    
    # Debug: save plot to project root as PNG
    try:
        import datetime as _datetime
        root_dir = os.path.dirname(os.path.abspath(__file__))
        filename = f"predict_plot_{_datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')}_{os.getpid()}.png"
        out_path = os.path.join(root_dir, filename)
        fig.savefig(out_path, format="png", bbox_inches="tight")
        print(f"[DEBUG] Saved prediction plot to {out_path}")
    except Exception as e:
        print(f"[WARN] Failed to save debug plot: {e}")
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    
    
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return b64


@app.get("/")
def index():
    return jsonify({
        "status": "ok",
        "message": "Send a POST multipart/form-data with field 'file' to /predict; POST JSON to /retrain to train a fresh model"
    })


@app.post("/predict")
def predict_endpoint():
    if 'file' not in request.files:
        return jsonify({"error": "No file part 'file' in request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        img = Image.open(file.stream).convert('RGB')
    except Exception as e:
        return jsonify({"error": f"Unable to open image: {e}"}), 400

    model, labels = load_model_and_labels()

    x, img_resized = preprocess_image(img)

    try:
        preds = model.predict(x, verbose=0)
    except Exception as e:
        return jsonify({"error": f"Model prediction failed: {e}"}), 500

    probs = preds[0].tolist()

    # Clip/normalize in case of numerical oddities
    probs = [float(max(0.0, min(1.0, p))) for p in probs]

    # Map probabilities to labels (truncate/extend to match)
    if len(probs) != len(labels):
        # Adjust lengths gracefully
        if len(probs) < len(labels):
            labels = labels[:len(probs)]
        else:
            probs = probs[:len(labels)]

    prob_map: Dict[str, float] = {label: float(p) for label, p in zip(labels, probs)}

    # Predicted label
    pred_idx = int(np.argmax(probs)) if probs else -1
    predicted_label = labels[pred_idx] if 0 <= pred_idx < len(labels) else None

    # Build plot
    plot_b64 = plot_prediction(img_resized, labels, probs)

    return jsonify({
        "predicted_label": predicted_label,
        "probabilities": prob_map,
        "plot_base64_png": plot_b64
    })


@app.post("/retrain")
def retrain_endpoint():
    global _model, _labels, MODEL_PATH, IMAGE_SIZE
    # Parse optional JSON body
    body: Dict[str, Any] = request.get_json(silent=True) or {}

    # Build TrainParams from provided fields with defaults
    params = TrainParams(
        data_dir=body.get('data_dir', TrainParams().data_dir),
        train_subdir=body.get('train_subdir', TrainParams().train_subdir),
        test_subdir=body.get('test_subdir', TrainParams().test_subdir),
        img_size=tuple(body.get('img_size', list(TrainParams().img_size))),
        batch_size=int(body.get('batch_size', TrainParams().batch_size)),
        epochs=int(body.get('epochs', TrainParams().epochs)),
        learning_rate=float(body.get('learning_rate', TrainParams().learning_rate)),
        base_model_name=str(body.get('base_model_name', TrainParams().base_model_name)),
        output_model_path=body.get('output_model_path', MODEL_PATH),
        per_class_limit=body.get('per_class_limit', TrainParams().per_class_limit),
        validation_split_from_test=float(body.get('validation_split_from_test', TrainParams().validation_split_from_test)),
    )

    try:
        result = train_model_service(params)
    except Exception as e:
        return jsonify({"error": f"Training failed: {e}"}), 500

    # Refresh in-memory model and labels for subsequent predictions
    try:
        MODEL_PATH = result.get('output_model_path', MODEL_PATH)
        _model = keras.models.load_model(MODEL_PATH)
        _labels = result.get('classes', _labels)
        # Also update IMAGE_SIZE used in preprocessing to match training
        img_size = result.get('used_params', {}).get('img_size')
        if img_size and isinstance(img_size, (list, tuple)) and len(img_size) == 2:
            IMAGE_SIZE = (int(img_size[0]), int(img_size[1]))
    except Exception as e:
        return jsonify({"warning": f"Model trained but failed to reload for inference: {e}", "details": result}), 200

    return jsonify({
        "message": "Model retrained successfully",
        "details": result
    })


if __name__ == "__main__":
    # Run the app
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)