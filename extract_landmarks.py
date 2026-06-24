import cv2
import mediapipe as mp
import csv
from pathlib import Path
import urllib.request

# This script extracts the mathematical coordinates for the 21 hand joints 
# and saves the data to csv files for training the dense neural network. 
class_names = [
    'Alapadma', 'Ardhapataka', 'Chandrakala', 'Hamsasyo', 'Kataka', 
    'Mukula', 'Mushti', 'Pataka', 'Shikara', 'Suchi'
]

# retrive mediapipe handtracking model
model_path = Path('hand_landmarker.task')
if not model_path.exists():
    print("Downloading the MediaPipe Hand Tracker model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, str(model_path))

#task apis for mediapipe to look at images, look for one hand and only accept detections with 50% accuracy
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
# create the hand tracking object
landmarker = HandLandmarker.create_from_options(options)

# This helper function handles the processing, based on a folder of images extracts the coordinates,
# and saves in a CSV file. 
def process_folder(folder_path, output_csv):
    folder_path = Path(folder_path)
    if not folder_path.exists():
        print(f"Error: Could not find {folder_path}!")
        return
    # csv in write mode to add daata
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        
        # spreadsheet formatting
        header = ['label']
        for i in range(21):
            header.extend([f'x{i}', f'y{i}', f'z{i}'])
        writer.writerow(header)

        # keep counts for debugging purposes
        print(f"\nProcessing {folder_path} -> {output_csv}...")
        success_count = 0
        fail_count = 0

        for class_idx, class_name in enumerate(class_names):
            class_folder = folder_path / class_name 
            if not class_folder.exists(): # throw error if folder not found
                print(f"  Skipping {class_name} - Folder not found.")
                continue
             # go through each picture in the folder
            for img_path in class_folder.glob('*.*'): 
                img = cv2.imread(str(img_path))
                if img is None: continue # skip broken images
                
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert rgb to bgr color format
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
                
                # detect the hand in the image
                results = landmarker.detect(mp_image)

                if results.hand_landmarks: # if a hand is found, add to csv
                    for hand_landmarks in results.hand_landmarks:
                        row = [class_idx]
                        for lm in hand_landmarks:
                            row.extend([lm.x, lm.y, lm.z])
                        writer.writerow(row)
                        success_count += 1
                else:
                    fail_count += 1
                    
        print(f"  Success: {success_count} images.")
        print(f"  Failed (no hand found): {fail_count} images.")

# MAIN CODE
# run the function on both the balanced training and testing datasets
process_folder('dataset/train_balanced', 'train_landmarks.csv')
process_folder('dataset/test_balanced', 'test_landmarks.csv')
print("\n-Extraction Complete.")