# Environment

Tested with Linux x86_64, Python 3.13.5 and NumPy 2.3.5. No GPU, CUDA, network service, API key, ffmpeg, OpenCV, pandas, PyTorch or scikit-learn is required by the top-level inference path.

Setup on a clean Linux host:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r environment/requirements.txt
```

Network access may be used during environment setup to obtain NumPy from PyPI. Inference itself requires no network access.
