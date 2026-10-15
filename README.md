# MediaPipe Hand Gesture Recognition

A real-time hand gesture recognition engine using Google's MediaPipe, OpenCV, and TensorFlow Lite. This system detects 21 hand landmarks and utilizes two lightweight neural networks to classify both static hand signs and dynamic finger movements.

## Project Structure
*   `app.py`: The main OpenCV and MediaPipe web-camera application.
*   `keypoint_classification.ipynb`: Jupyter Notebook to train the static hand sign recognition model.
*   `point_history_classification.ipynb`: Jupyter Notebook to train the dynamic gesture recognition model.
*   `model/`: Contains the trained `.tflite` and `.hdf5` models, the coordinate datasets (`.csv`), and the Python wrapper classes for inference.
*   `utils/`: Contains the `cvfpscalc.py` utility script to calculate the video frame rate.

## Setup & Installation
It is recommended to run this in a virtual environment. Install the required dependencies using:
```bash
pip install -r requirements.txt

## Usage
1. Run the Live Recognition App
Bash
python app.py
ESC: Exit the application.

n: Normal mode (recognition only).

k: Static Sign Logging Mode. Press numbers 0-9 to log the current hand pose coordinates to model/keypoint_classifier/keypoint.csv.

h: Dynamic Gesture Logging Mode. Press numbers 0-9 to log the current finger movement history to model/point_history_classifier/point_history.csv.

2. Train Custom Gestures
To add your own gestures, use the live application to log new data via the k or h modes. Then, open keypoint_classification.ipynb or point_history_classification.ipynb and run all cells. The notebooks will automatically retrain the Multi-Layer Perceptron (MLP) and export the updated .tflite models for immediate use in app.py.
