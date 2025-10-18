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
   curl -X POST \
        -F "file=@/path/to/your/image.jpg" \
        http://localhost:8000/predict

Example JSON response (fields abbreviated):
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

The plot is also saved to a debug PNG in the project root (ignored by Git) and printed in the server logs as:
[DEBUG] Saved prediction plot to /path/to/repo/predict_plot_YYYYmmdd-HHMMSS-ffffff_PID.png

## API
- GET /
  - Health/info endpoint. Returns a small JSON with usage info.
- POST /predict
  - Multipart/form-data upload. Field name: file
  - Returns JSON with predicted_label, probabilities, plot_base64_png

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
