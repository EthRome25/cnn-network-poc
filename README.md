# cnn-network-poc
PoC neural network for detecting features based on the medical images

## Generating requirements.txt from the notebook
This repo includes a helper script to generate a requirements file from imports in the Jupyter notebook.

Usage:
- Generate requirements.txt in the current directory:
  - python generate_requirements.py brain-tumor-mri-accuracy-99.ipynb
- Or write to a specific output path:
  - python generate_requirements.py brain-tumor-mri-accuracy-99.ipynb -o requirements.txt

The script parses imports, maps common module names to PyPI packages (e.g., PIL -> Pillow, sklearn -> scikit-learn), and excludes standard library modules. Versions are left unpinned by design.
