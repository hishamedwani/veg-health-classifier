# 🌱 Veg Health Classifier

Classifies potato, tomato, and bell pepper leaf photos by crop type and health condition (healthy or which disease), using a MobileNetV2 transfer-learning model trained on the PlantVillage dataset.

## Features

- Upload a photo, or capture one live from your laptop camera
- Identifies the vegetable (potato / tomato / bell pepper)
- Flags healthy vs. the specific disease detected, with a confidence score

## Model Details

- Architecture: MobileNetV2 (transfer learning, fine-tuned)
- Dataset: PlantVillage (color images), filtered to Potato/Tomato/Pepper classes
- Framework: TensorFlow / Keras

## Technologies

- **Backend:** Flask
- **ML Model:** TensorFlow / Keras (MobileNetV2)
- **Deployment:** Docker
- **Hosting:** Render

## Local Setup

### Prerequisites

- Python 3.11+
- Docker

### Run with Docker

```
docker build -t veg-health-classifier .
docker run -p 5000:5000 veg-health-classifier
```

Open browser: <http://localhost:5000>

## Training

The model is trained in `train_veg_classifier.ipynb` on Google Colab (GPU recommended). It downloads the PlantVillage dataset via the Kaggle API, filters to Potato/Tomato/Pepper classes, and fine-tunes a MobileNetV2. Outputs `veg_classifier.h5` and `class_names.json`, which belong in the repo root alongside `app.py`.

## How to Use

1. Choose "Upload" or "Camera" mode
2. Provide a leaf photo
3. Click "Classify"
4. View the vegetable, health status, and confidence score

## Known Limitations

- Best performance with clear, well-lit, close-up leaf photos
- Only recognizes Potato, Tomato, and Bell Pepper — other crops will be misclassified into one of these
- Trained on PlantVillage's lab-style backgrounds; real-world/field photos may reduce accuracy

## Future Improvements

- Support for more crops
- Bounding-box localization for multi-leaf images
