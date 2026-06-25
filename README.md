# Mudra Magic
This application recognizes and classifies 10 different hand gestures used in classical dance.
These 10 are pataka, ardhapataka, alapadma, chandrakala, hamsasyo, kataka, mukula, mushti, shikara, and suchi. 

## Run this project
Use the following command:
```bash
python webcam_landmarks.py
```

If the csv and keras files are corrupted:
```bash
 python extract_landmarks.py
 python train_landmarks.py
 python webcam_landmarks.py
```

To balance the dataset:
```bash
$ balance_data.py
```

## Sources
- Dataset, says balanced but is not: https://www.kaggle.com/datasets/krithi9977/bharatanatyam-mudra-dataset-balanced/versions/1?resource=download 
- https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker?authuser=1 
- https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense 
- https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html 
- https://www.w3schools.com/python/ 
- inspiration to apply similar concept to indian classical dance: https://blogs.nvidia.com/blog/ai-sign-language/ 
