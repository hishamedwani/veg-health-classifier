import io
import json
import os

import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "veg_classifier.keras")
CLASSES_PATH = os.path.join(os.path.dirname(__file__), "class_names.json")
IMG_SIZE = (224, 224)

model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASSES_PATH) as f:
    CLASS_NAMES = json.load(f)


def parse_label(raw_label):
    """
    PlantVillage labels look like 'Potato___Early_blight' or
    'Pepper__bell___healthy'. Split into (vegetable, condition).
    """
    veg_map = {
        "Potato": "Potato",
        "Tomato": "Tomato",
        "Pepper": "Bell Pepper",
    }
    veg_key = raw_label.split("_")[0]
    vegetable = veg_map.get(veg_key, veg_key)

    condition_raw = raw_label.split("___")[-1] if "___" in raw_label else raw_label
    condition = condition_raw.replace("_", " ").strip()
    is_healthy = condition.lower() == "healthy"

    return vegetable, condition, is_healthy


def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image_bytes = file.read()

    try:
        arr = preprocess_image(image_bytes)
    except Exception:
        return jsonify({"error": "Could not read image"}), 400

    preds = model.predict(arr)[0]
    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx])
    raw_label = CLASS_NAMES[top_idx]

    vegetable, condition, is_healthy = parse_label(raw_label)

    # Return top 3 for a bit of transparency in the UI
    top3_idx = preds.argsort()[-3:][::-1]
    top3 = [
        {
            "label": CLASS_NAMES[i],
            "confidence": float(preds[i]),
        }
        for i in top3_idx
    ]

    return jsonify({
        "vegetable": vegetable,
        "condition": condition,
        "is_healthy": is_healthy,
        "confidence": round(confidence * 100, 2),
        "top3": top3,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
