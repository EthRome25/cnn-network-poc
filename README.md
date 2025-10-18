# CNN Network PoC — Brain Tumor MRI Classifier (HTTP API)

This repository contains a simple Flask web server that loads a trained Keras model (trained-model.keras) and exposes a single unauthenticated HTTP endpoint to run predictions on uploaded images. The server returns:
- predicted_label: the most probable class
- probabilities: per-class probabilities
- plot_base64_png: a base64-encoded PNG of a matplotlib figure showing the input image and a horizontal bar chart of probabilities

Additionally, each request saves the same plot as a PNG in the project root for debugging (files named predict_plot_YYYYmmdd-HHMMSS-ffffff_PID.png). These files are ignored by Git.

## Project layout
- app.py — Flask server with / and /predict endpoints
- trained-model.keras — Saved Keras model file (must be present or provided via MODEL_PATH)
- requirements.txt — Python dependencies (TensorFlow, Flask, etc.)
- brain-tumor-mri-accuracy-99-cpu-lite.ipynb — The training/evaluation notebook used to produce the model
- input_data/ — Example dataset (ignored by default in .gitignore)

## Prerequisites
- Python 3.9–3.11 recommended
- pip
- A CPU build of TensorFlow is sufficient. GPU is optional.

Tip for Apple Silicon (M1/M2/M3): installing TensorFlow may require a Conda environment or a specific wheel. If pip install -r requirements.txt fails, consult TensorFlow’s official install guide for your platform.

## Quick start (run locally)
1) Clone the repo and enter the folder
   git clone <this-repo-url>
   cd cnn-network-poc

2) (Optional) Create and activate a virtual environment
   python -m venv .venv
   source .venv/bin/activate    # macOS/Linux
   .venv\\Scripts\\activate       # Windows PowerShell

3) Install dependencies
   pip install -r requirements.txt

4) Provide the model file
   - Option A: Place trained-model.keras in the project root (same directory as app.py).
   - Option B: Keep the model elsewhere and set an environment variable to point to it:
     export MODEL_PATH=/absolute/path/to/trained-model.keras

5) Start the server
   python app.py
   # The app listens on 0.0.0.0:8000 by default. To change the port:
   PORT=5000 python app.py

6) Make a prediction request (example using curl)
```shell
   curl -X POST \
        -F "file=@/path/to/your/image.jpg" \
        http://localhost:8000/predict
```

Example JSON response (fields abbreviated):
```json
{
  "predicted_label": "meningioma",
  "probabilities": {
    "glioma": 0.02,
    "meningioma": 0.93,
    "notumor": 0.01,
    "pituitary": 0.04
  },
  "plot_base64_png": "iVBORw0KGgoAAA..."
}
```

The plot is also saved to a debug PNG in the project root (ignored by Git) and printed in the server logs as:
[DEBUG] Saved prediction plot to /path/to/repo/predict_plot_YYYYmmdd-HHMMSS-ffffff_PID.png

## API
- GET /
  - Health/info endpoint. Returns a small JSON with usage info.
- POST /predict
  - Multipart/form-data upload. Field name: file
  - Returns JSON with predicted_label, probabilities, plot_base64_png
- POST /retrain
  - JSON body with optional parameters to train a new model from scratch on the dataset and overwrite the saved model.
  - Optional JSON fields (all optional):
    - data_dir: string path to dataset root (default: ./input_data)
    - train_subdir: subfolder name for training data (default: Training)
    - test_subdir: subfolder name for test data (default: Testing)
    - img_size: [width, height] (default: [128, 128])
    - batch_size: integer (default: 8)
    - epochs: integer (default: 6)
    - learning_rate: float (default: 0.001)
    - base_model_name: one of MobileNetV2, EfficientNetB0 (default: MobileNetV2)
    - output_model_path: where to save the trained model (default: ./trained-model.keras)
    - per_class_limit: integer limit per class for quick CPU training; null/omitted means use all data (default: null)
    - validation_split_from_test: fraction of the Testing dir to use as validation vs. test (default: 0.5)
  - Response JSON contains training history arrays, final epoch, evaluation scores on train/valid/test, classes, and path of the saved model. The server reloads the new model for subsequent /predict calls.

### Example retrain request
```shell
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
        "epochs": 6,
        "batch_size": 8,
        "per_class_limit": null
      }' \
  http://localhost:8000/retrain
```

### Example retrain response (truncated)
```json
{
    "details": {
        "classes": [
            "glioma",
            "meningioma",
            "notumor",
            "pituitary"
        ],
        "final_epoch": 6,
        "history": {
            "accuracy": [
                0.37187498807907104,
                0.581250011920929,
                0.703125,
                0.7406250238418579,
                0.8031250238418579,
                0.828125
            ],
            "loss": [
                1.4224234819412231,
                0.9574562907218933,
                0.7831178307533264,
                0.7211284637451172,
                0.5772733092308044,
                0.5526394248008728
            ],
            "precision": [
                0.4424242377281189,
                0.6751269102096558,
                0.7848101258277893,
                0.8132780194282532,
                0.8689138293266296,
                0.8713235259056091
            ],
            "recall": [
                0.22812500596046448,
                0.4156250059604645,
                0.581250011920929,
                0.612500011920929,
                0.7250000238418579,
                0.7406250238418579
            ],
            "val_accuracy": [
                0.4749999940395355,
                0.6000000238418579,
                0.6625000238418579,
                0.6875,
                0.699999988079071,
                0.699999988079071
            ],
            "val_loss": [
                1.1323192119598389,
                0.9522106051445007,
                0.8516246676445007,
                0.7921577095985413,
                0.7443216443061829,
                0.7126464247703552
            ],
            "val_precision": [
                0.6585366129875183,
                0.7222222089767456,
                0.75,
                0.7272727489471436,
                0.7692307829856873,
                0.7647058963775635
            ],
            "val_recall": [
                0.3375000059604645,
                0.48750001192092896,
                0.5625,
                0.6000000238418579,
                0.625,
                0.6499999761581421
            ]
        },
        "output_model_path": "/trained-model.keras",
        "test_score": {
            "compile_metrics": 0.7749999761581421,
            "loss": 0.5866514444351196
        },
        "train_score": {
            "compile_metrics": 0.859375,
            "loss": 0.46055832505226135
        },
        "used_params": {
            "base_model_name": "MobileNetV2",
            "batch_size": 8,
            "data_dir": "/input_data",
            "epochs": 6,
            "img_size": [
                128,
                128
            ],
            "learning_rate": 0.001,
            "output_model_path": "/trained-model.keras",
            "per_class_limit": 80,
            "test_subdir": "Testing",
            "train_subdir": "Training",
            "validation_split_from_test": 0.5
        },
        "valid_score": {
            "compile_metrics": 0.699999988079071,
            "loss": 0.7126464247703552
        }
    },
    "message": "Model retrained successfully"
}
```

### Example GET / response
```json
{
  "status": "ok",
  "message": "Send a POST multipart/form-data with field 'file' to /predict; POST JSON to /retrain to train a fresh model"
}
```

Notes:
- Training is CPU-intensive and may take several minutes depending on the dataset and parameters.
- This endpoint runs synchronously; the HTTP request remains open until training completes. Consider using a process manager or async job queue for production.

## Configuration
- MODEL_PATH — Optional. Full path to a .keras model file. Defaults to ./trained-model.keras.
- PORT — Optional. Port to bind the Flask server. Default: 8000.

## Troubleshooting
- TensorFlow install issues: ensure your Python version is supported. On Apple Silicon, consider a Conda environment if pip wheels are problematic.
- Large model file: make sure MODEL_PATH points to the right file location.
- 400 "Unable to open image": verify the uploaded file is a valid image (JPEG/PNG) and the form field key is exactly file.
- 500 "Model prediction failed": check model compatibility with the input size (128x128 RGB) and that TensorFlow can load the model.

## License
This project is licensed under the terms of the LICENSE file found in this repository.
