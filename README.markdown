# In-Demand Machine Learning Models for 2025

This repository contains six machine learning models that are highly demanded in 2025, covering tabular data, computer vision, NLP, and time-series tasks.

## Models Included

1. **XGBoost Classifier** (`xgboost_classifier.py`): Classifies tennis swing types using tabular data.
2. **ResNet-50 Image Classifier** (`resnet_classifier.py`): Classifies images using a pre-trained ResNet-50 model.
3. **BERT Text Classifier** (`bert_classifier.py`): Performs sentiment analysis on text using BERT.
4. **Random Forest Classifier** (`random_forest_classifier.py`): Detects fraud using tabular data.
5. **LSTM Network** (`lstm_forecasting.py`): Forecasts energy demand using time-series data.
6. **YOLOv8 Object Detection** (`yolov8_object_detection.py`): Performs real-time object detection on images.

## Prerequisites

- Python 3.8+

- Install dependencies:

  ```bash
  pip install pandas xgboost scikit-learn joblib torch torchvision transformers pillow numpy tensorflow matplotlib ultralytics opencv-python
  ```

## Dataset and Files

- **swing_data.csv**: Required for `xgboost_classifier.py`. Format: columns `accelX`, `accelY`, `accelZ`, `gyroX`, `gyroY`, `gyroZ`, `swingType`.
- **fraud_data.csv**: Required for `random_forest_classifier.py`. Format: features and a label column `is_fraud`.
- **energy_demand.csv**: Required for `lstm_forecasting.py`. Format: a column `demand` with time-series data.
- **example_image.jpg**: Required for `resnet_classifier.py` and `yolov8_object_detection.py`. Replace with your image.
- **imagenet_classes.txt**: Required for `resnet_classifier.py`. Download from ImageNet or similar sources.

## Usage

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Run each script:

   - XGBoost: `python xgboost_classifier.py`
   - ResNet: `python resnet_classifier.py`
   - BERT: `python bert_classifier.py`
   - Random Forest: `python random_forest_classifier.py`
   - LSTM: `python lstm_forecasting.py`
   - YOLOv8: `python yolov8_object_detection.py`

## Notes

- Ensure you have a GPU for faster inference with ResNet, BERT, LSTM, and YOLOv8 models.
- Replace placeholder files (e.g., `example_image.jpg`) with your data.
- YOLOv8 requires internet access to download the pre-trained weights (`yolov8n.pt`) on first run.

## License

MIT License