# Mudra Magic

> A real-time hand-gesture recognition application that classifies 10 Bharatanatyam mudras using computer vision, hand-landmark extraction, and a neural network.

Mudra Magic was developed to explore how computer vision and machine learning can be applied to Indian classical dance. As a Bharatanatyam dancer, I wanted to combine my interests into developing a unique application. 

The application uses webcam input to detect hand landmarks and classify 10 Bharatanatyam hand gestures in real time. The project combines MediaPipe-based landmark extraction, OpenCV video processing, and a Keras neural network to translate hand position and structure into a predicted mudra.

## The challenge

Bharatanatyam relies heavily on precise hand gestures, or mudras, to communicate meaning and expression.

Recognizing these gestures computationally can be challenging because different mudras may have subtle differences in finger position, orientation, and hand shape.

Mudra Magic was designed to create a lightweight recognition pipeline that could:

- Detect a dancer's hand from live webcam input
- Extract meaningful hand-landmark features
- Distinguish between visually similar mudras
- Classify gestures in real time
- Provide a foundation for future dance-learning or movement-analysis tools

## Our solution

Mudra Magic uses a landmark-based classification approach rather than training directly on raw images.

The system processes each frame through a hand-landmark detector, converts the detected hand structure into numerical features, and passes those features into a trained neural network.

The general pipeline is:

```text
Webcam Input
     |
     v
Hand Detection
     |
     v
MediaPipe Hand Landmarks
     |
     v
Landmark Feature Extraction
     |
     v
Neural Network Classifier
     |
     v
Predicted Bharatanatyam Mudra
```

This approach reduces the amount of visual information the classifier must process and focuses the model on the relative structure of the hand.

## Supported mudras

The current model recognizes 10 single-handed gestures used in Bharatanatyam:

- Pataka
- Ardhapataka
- Alapadma
- Chandrakala
- Hamsasyo
- Kataka
- Mukula
- Mushti
- Shikara
- Suchi

## Features

- **Real-time gesture recognition** — Uses webcam input to classify Bharatanatyam mudras as they are performed.
- **Hand-landmark extraction** — Uses MediaPipe to identify key hand points and convert gestures into structured numerical features.
- **Neural-network classification** — Uses a Keras-based dense neural network to classify extracted landmarks.
- **10-class prediction** — Distinguishes between 10 predefined Bharatanatyam mudras.
- **Dataset preprocessing** — Includes scripts for extracting landmark data and preparing it for model training.
- **Dataset balancing** — Includes a utility for balancing training examples across gesture classes.
- **Retrainable pipeline** — The model can be rebuilt from the source dataset if generated CSV or Keras files need to be recreated.

## Technology

- Python
- OpenCV
- MediaPipe
- TensorFlow
- Keras
- Dense Neural Networks
- Computer Vision
- Hand Landmark Detection
- Machine Learning
- CSV Data Processing

## Getting started

Clone the repository:

```bash
git clone https://github.com/dishar-proj/MudraMagic.git
cd MudraMagic
```

Run the real-time classifier:

```bash
python webcam_landmarks.py
```

The application will open the webcam and begin detecting and classifying supported hand gestures.

## Rebuilding the model

If the generated landmark CSV or Keras model files are unavailable or need to be recreated, run:

```bash
python extract_landmarks.py
python train_landmarks.py
python webcam_landmarks.py
```

The workflow is:

```text
Image Dataset
     |
     v
extract_landmarks.py
     |
     v
Landmark Dataset
     |
     v
train_landmarks.py
     |
     v
Trained Keras Model
     |
     v
webcam_landmarks.py
```

## Balancing the dataset

The original dataset may contain an unequal number of examples across gesture classes.

To generate a more balanced dataset, run:

```bash
python balance_data.py
```

This helps reduce class imbalance during model training.

## Project structure

```text
MudraMagic/
├── webcam_landmarks.py      # Real-time webcam recognition
├── extract_landmarks.py     # Extracts hand landmarks from image data
├── train_landmarks.py       # Trains the landmark-based classifier
├── balance_data.py          # Balances examples across mudra classes
├── *.csv                    # Extracted landmark data
├── *.keras                  # Trained Keras model
└── README.md
```

## How it works

### Hand detection

Each webcam frame is processed using MediaPipe's hand-landmark detection tools.

Rather than using the entire image as the model input, the application extracts the detected hand's landmark coordinates.

### Feature extraction

The landmark coordinates provide a structured representation of the shape and position of the hand.

These features are used as the input to the classifier.

### Classification

A dense neural network implemented with TensorFlow/Keras predicts which of the supported Bharatanatyam mudras most closely matches the detected hand configuration.

### Real-time prediction

OpenCV handles the webcam stream and allows the hand-recognition pipeline to continuously process frames and display predictions.

## Dataset

The project uses a publicly available Bharatanatyam mudra image dataset as the starting point for training.

Dataset:

[Bharatanatyam Mudra Dataset — Kaggle](https://www.kaggle.com/datasets/krithi9977/bharatanatyam-mudra-dataset-balanced/versions/1?resource=download)

Although the dataset is described as balanced, additional balancing was performed as part of this project to improve the distribution of training examples across classes.

## Applications

Mudra Magic demonstrates how computer vision can be used to interpret structured human movement.

Potential future applications include:

- Interactive Bharatanatyam learning tools
- Real-time gesture feedback for dancers
- Dance-form digitization and preservation
- Movement-analysis systems
- Educational tools for identifying mudras
- Gesture-based interfaces inspired by Indian classical dance

## Future improvements

Future versions could expand the project through:

- Recognition of additional Bharatanatyam mudras
- Detection of both hands simultaneously
- Sequence recognition for complete dance phrases
- Confidence-score visualization
- Improved robustness across lighting conditions and camera angles
- Personalized feedback for hand positioning
- Integration into a web or mobile learning experience
- Larger and more diverse training datasets

## Important note

Mudra Magic is an educational machine-learning project and prototype.

Model predictions may vary depending on lighting, camera position, hand orientation, and differences between the training data and the user performing the gesture.

The application is not intended to replace formal instruction in Bharatanatyam.

## References

- [Bharatanatyam Mudra Dataset — Kaggle](https://www.kaggle.com/datasets/krithi9977/bharatanatyam-mudra-dataset-balanced/versions/1?resource=download)
- [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker?authuser=1)
- [TensorFlow Keras Dense Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)
- [OpenCV Video Capture](https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html)
- [Python Reference — W3Schools](https://www.w3schools.com/python/)
- [NVIDIA: Applying AI to Sign Language Recognition](https://blogs.nvidia.com/blog/ai-sign-language/)

## Acknowledgments

Mudra Magic was created to explore the intersection of computer vision, machine learning, and Indian classical dance.

The project applies modern gesture-recognition techniques to Bharatanatyam with the goal of exploring how technology can support the learning, analysis, and preservation of traditional movement systems.
