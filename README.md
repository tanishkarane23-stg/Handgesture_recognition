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
