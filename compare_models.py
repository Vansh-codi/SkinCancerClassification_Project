import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
import time
import h5py
import json
import zipfile
import os
import shutil
import tempfile
# Helper: strip quantization_config

def strip_quantization_config(config):
    if isinstance(config, dict):
        config.pop('quantization_config', None)
        for v in config.values():
            strip_quantization_config(v)
    elif isinstance(config, list):
        for item in config:
            strip_quantization_config(item)
    return config

# Helper: load .h5 safely
def load_h5_model_safe(path):
    # with h5py.File(path, 'r+') as f:
    with h5py.File(path, 'r+') as f:
        model_config = json.loads(f.attrs['model_config'])
        model_config = strip_quantization_config(model_config)
        f.attrs['model_config'] = json.dumps(model_config)
    return tf.keras.models.load_model(path, compile=False, safe_mode=False)

# Helper: load .keras safely

def load_keras_model_safe(path):
    tmpdir = tempfile.mkdtemp()
    with zipfile.ZipFile(path, 'r') as z:
        z.extractall(tmpdir)

    config_path = os.path.join(tmpdir, 'config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    config = strip_quantization_config(config)

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f)

    fixed_path = path.replace('.keras', '_fixed.keras')
    with zipfile.ZipFile(fixed_path, 'w', zipfile.ZIP_DEFLATED) as zout:
        for root, dirs, files in os.walk(tmpdir):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, tmpdir)
                zout.write(full_path, arcname)

    shutil.rmtree(tmpdir)
    return tf.keras.models.load_model(fixed_path, compile=False)

# Load all 3 models

print("Loading models...")

try:
    model1 = load_h5_model_safe("models/skin_cancer_model.h5")
    print("Model 1 loaded ✓  (Basic CNN | 180x180 | 25 epochs)")
except Exception as e:
    print(f"Model 1 failed: {e}")
    model1 = None

try:
    model2 = load_h5_model_safe("models/improved_skin_cancer_model.h5")
    print("Model 2 loaded ✓  (Improved CNN | 224x224 | 40 epochs)")
except Exception as e:
    print(f"Model 2 failed: {e}")
    model2 = None

try:
    model3 = load_keras_model_safe("models/final_skin_cancer_model.keras")
    print("Model 3 loaded ✓  (Final CNN + Callbacks | 224x224 | 40 epochs)")
except Exception as e:
    print(f"Model 3 failed: {e}")
    model3 = None

# Class names (HAM10000 - 9 classes)

class_names = [
    'actinic keratosis',
    'basal cell carcinoma',
    'dermatofibroma',
    'melanoma',
    'nevus',
    'pigmented benign keratosis',
    'seborrheic keratosis',
    'squamous cell carcinoma',
    'vascular lesion'
]

# Helper: get model input size

def get_input_size(model):
    shape = model.input_shape  # (None, H, W, 3)
    return (shape[1], shape[2])
# Helper: predict on image
# Model 1 has no built-in Rescaling → divide by 255
# Model 2 has no built-in Rescaling → divide by 255
# Model 3 has built-in Rescaling layer → no division needed
def predict_model(model, name, model_num, image_path):
    h, w = get_input_size(model)
    img = load_img(image_path, target_size=(h, w))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # (1, H, W, 3)

    # Manual normalization for models without built-in Rescaling
    if model_num in [1, 2]:
        img_array = img_array / 255.0

    start = time.time()
    pred = model.predict(img_array, verbose=0)
    elapsed = time.time() - start

    class_idx = np.argmax(pred)
    class_name = class_names[class_idx]
    confidence = float(np.max(pred)) * 100

    return class_name, confidence, elapsed, h, w
# Image path
IMAGE_PATH = "images/sample_image.jpg"

if not os.path.exists(IMAGE_PATH):
    print(f"\nERROR: '{IMAGE_PATH}' not found in {os.getcwd()}")
    print("Please place a skin lesion image here:")
    print(os.path.join(os.getcwd(), "images"))
    # print("Please place a skin lesion image named 'sample_image.jpg' in:")
    # print(f"  {os.getcwd()}")
    exit()
# Run predictions
print(f"\nTesting on image: {IMAGE_PATH}")
print("=" * 65)
print(f"{'Model':<30} {'Prediction':<28} {'Conf':>7} {'Time':>8}")
print("-" * 65)

results = {}
models = [
    (model1, "Model 1 — Basic CNN",          1),
    (model2, "Model 2 — Improved CNN",        2),
    (model3, "Model 3 — Final + Callbacks",   3),
]

for model, name, num in models:
    if model:
        pred, conf, t, h, w = predict_model(model, name, num, IMAGE_PATH)
        results[name] = (pred, conf, t, h, w)
        print(f"{name:<30} {pred:<28} {conf:>6.1f}% {t:>7.4f}s")
    else:
        print(f"{name:<30} {'FAILED TO LOAD':<28}")

# Summary with training stats

print("\n" + "=" * 65)
print("EXPERIMENT SUMMARY")
print("=" * 65)

summary = [
    ("Model 1 — Basic CNN",        "180x180", "25",          "~55%",  "~39%"),
    ("Model 2 — Improved CNN",     "224x224", "40",          "~62%",  "~39%"),
    ("Model 3 — Final+Callbacks",  "224x224", "40+callbacks","~56%",  "~33%"),
]

print(f"{'Model':<30} {'Input':>8} {'Epochs':>14} {'TrainAcc':>10} {'ValAcc':>8}")
print("-" * 75)
for row in summary:
    print(f"{row[0]:<30} {row[1]:>8} {row[2]:>14} {row[3]:>10} {row[4]:>8}")

print("\nConclusion:")
print("  All 3 models trained on HAM10000 (9 classes, same dataset).")
print("  Model 2 achieved highest train accuracy (62%).")
print("  Callbacks in Model 3 prevented overfitting but restricted learning.")
print("  Val accuracy plateau at ~39% suggests transfer learning is needed.")
print("\nDone.")