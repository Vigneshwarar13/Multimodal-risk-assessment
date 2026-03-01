modules = [
    "numpy",
    "scipy",
    "pandas",
    "cv2",
    "mediapipe",
    "tensorflow",
    "torch",
    "torchvision",
    "transformers",
    "sentencepiece",
    "tokenizers",
    "whisper",
    "librosa",
    "soundfile",
    "streamlit",
    "plotly",
    "sklearn",
    "matplotlib",
    "seaborn",
    "fastapi",
    "uvicorn",
    "ffmpeg",
    "sounddevice",
    "yaml",
    "joblib",
    "sentencepiece",
]

import importlib
import sys

results = []
for m in modules:
    try:
        importlib.import_module(m)
        results.append((m, True, ""))
    except Exception as e:
        results.append((m, False, str(e)))

for name, ok, err in results:
    if ok:
        print(f"OK: {name}")
    else:
        print(f"MISSING: {name} -> {err}")

sys.exit(0)
