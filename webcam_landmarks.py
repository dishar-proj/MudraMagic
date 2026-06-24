import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from pathlib import Path

# This script uses the webcam for a custom window and mediapipe to process 
# live input and shows mudra classification on the window. 

# Format the UI
# UI colors in BGR format
BANNER_BG_COLOR = (58, 71, 157)     #deep red
PANEL_BG_COLOR  = (30, 30, 30)      #gray
TEXT_COLOR_MAIN = (255, 255, 255)   #white
TEXT_COLOR_ACCENT = (68, 160, 222)  # yellow

# Text fonts and sizes
TITLE_FONT = cv2.FONT_HERSHEY_COMPLEX
BODY_FONT  = cv2.FONT_HERSHEY_SIMPLEX
BANNER_HEIGHT = 80
PANEL_WIDTH = 350

#Names of 10 mudras
class_names = [ 'Alapadma', 'Ardhapataka', 'Chandrakala', 'Hamsasyo', 'Kataka', 
    'Mukula', 'Mushti', 'Pataka', 'Shikara', 'Suchi']

#Meanings for each of the 10 mudras to display on the screen (got from my dance teacher)
mudra_meanings = {
    'Alapadma': "Fully bloomed lotus flower.",
    'Ardhapataka': "Half of a flag.",
    'Chandrakala': "Crescent moon.",
    'Hamsasyo': "Swan beak.",
    'Kataka': "Opening a bracelet.",
    'Mukula': "Flower blossom.",
    'Mushti': "Fist.",
    'Pataka': "Flag.",
    'Shikara': "A peak.",
    'Suchi': "A needle."
}

#AI hand tracking section------------------------------------------------------

#get the trained AI model generated previously
model = tf.keras.models.load_model('landmark_mudra_model_10.keras')

#open MediaPipe to get mappings and live video
model_path = Path('hand_landmarker.task')
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(model_path)),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5
)
landmarker = HandLandmarker.create_from_options(options)

#skeletion layout defining joints and fingers to show the green skeleton
CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),         # Thumb
    (0,5), (5,6), (6,7), (7,8),         # Index
    (5,9), (9,10), (10,11), (11,12),    # Middle
    (9,13), (13,14), (14,15), (15,16),  # Ring
    (13,17), (0,17), (17,18), (18,19), (19,20) # Pinky
]

#text wrapping the meanings incase they are too long neatly
def draw_wrapped_text(img, text, x, y, font, scale, color, thickness, max_width):
    words = text.split(' ')
    line = ""
    for word in words:
        test_line = line + word + " "
        size = cv2.getTextSize(test_line, font, scale, thickness)[0]
        if size[0] > max_width:
            cv2.putText(img, line, (x, y), font, scale, color, thickness, cv2.LINE_AA)
            y += int(size[1] * 1.8) # Move down for the next line
            line = word + " "
        else:
            line = test_line
    cv2.putText(img, line, (x, y), font, scale, color, thickness, cv2.LINE_AA)

#Webcam section----------------------------------------------------------------
cap = cv2.VideoCapture(0) # open default camera

while cap.isOpened():
    success, frame = cap.read() # read one video 
    if not success: continue

    #mirror the camera
    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    #Dashboard
    dashboard = np.zeros((H + BANNER_HEIGHT, W + PANEL_WIDTH, 3), dtype=np.uint8)
    
    # Draw Banner Background
    dashboard[0:BANNER_HEIGHT, 0:W + PANEL_WIDTH] = BANNER_BG_COLOR
    
    # Draw Panel Background
    dashboard[BANNER_HEIGHT:H + BANNER_HEIGHT, W:W + PANEL_WIDTH] = PANEL_BG_COLOR
    
    # Add Title to Banner
    cv2.putText(dashboard, "MUDRA MAGIC", (20, 55), TITLE_FONT, 1.5, TEXT_COLOR_MAIN, 2, cv2.LINE_AA)

    #AI tracking
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = landmarker.detect(mp_image)

    # Default UI states if no hand is detected
    current_mudra = "Waiting..."
    current_confidence = ""
    current_meaning = "Place your hand in view of the camera to begin translation."

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            
            # Draw Skeleton
            for start_idx, end_idx in CONNECTIONS:
                start_lm = hand_landmarks[start_idx]
                end_lm = hand_landmarks[end_idx]
                start_point = (int(start_lm.x * W), int(start_lm.y * H))
                end_point = (int(end_lm.x * W), int(end_lm.y * H))
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
            
            coords = []
            for lm in hand_landmarks:
                cx, cy = int(lm.x * W), int(lm.y * H)
                cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)
                coords.extend([lm.x, lm.y, lm.z])
            
            # Predict the Mudra
            input_data = np.array([coords])
            predictions = model.predict(input_data, verbose=0)
            predicted_index = np.argmax(predictions)
            confidence = np.max(predictions)
            
            if confidence > 0.50:
                current_mudra = class_names[predicted_index]
                current_confidence = f"Confidence: {confidence*100:.1f}%"
                current_meaning = mudra_meanings.get(current_mudra, "Meaning unknown.")

    # add live data to the side panel
    panel_start_x = W + 20
    panel_start_y = BANNER_HEIGHT + 40
    
    # Draw Gesture Name
    cv2.putText(dashboard, "Detected Mudra:", (panel_start_x, panel_start_y), BODY_FONT, 0.6, TEXT_COLOR_ACCENT, 1, cv2.LINE_AA)
    cv2.putText(dashboard, current_mudra, (panel_start_x, panel_start_y + 40), TITLE_FONT, 1.2, TEXT_COLOR_MAIN, 2, cv2.LINE_AA)
    
    # Draw Confidence Score
    cv2.putText(dashboard, current_confidence, (panel_start_x, panel_start_y + 80), BODY_FONT, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    
    # Draw Meaning text (Using our custom wrapper so it doesn't run off the screen)
    cv2.putText(dashboard, "Traditional Meaning:", (panel_start_x, panel_start_y + 160), BODY_FONT, 0.6, TEXT_COLOR_ACCENT, 1, cv2.LINE_AA)
    draw_wrapped_text(dashboard, current_meaning, panel_start_x, panel_start_y + 200, BODY_FONT, 0.7, TEXT_COLOR_MAIN, 1, PANEL_WIDTH - 40)

    # add webcam feed to dashboard
    dashboard[BANNER_HEIGHT:H + BANNER_HEIGHT, 0:W] = frame

    cv2.imshow('Mudra Magic AI Dashboard', dashboard)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()