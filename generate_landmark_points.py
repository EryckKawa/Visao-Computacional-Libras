import mediapipe as mp 
from mediapipe.tasks import python 
from mediapipe.tasks.python import vision
import cv2
import matplotlib.pyplot as plt

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

import pickle

import os

model_pose_path = './tasks/pose_landmarker_heavy.task'
model_hand_path = './tasks/hand_landmarker.task'

##############################################################

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

## CRIACAO DE POSE LANDMARKER

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

# Create a pose landmarker instance with the video mode:
options_pose = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_pose_path),
    running_mode=VisionRunningMode.VIDEO)

## CRIACAO DE HAND LANDMARKER

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

# Create a hand landrmarker instance with the video mode:
options_hand = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_hand_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2)

def get_filenames():
    # Path to the folder containing the files
    folder_path = "./DB"

    # List to store filenames without .mp4 extension
    filenames = []

    # Get all files in the folder
    for file in os.listdir(folder_path):
        if file.endswith(".mp4"):
            filenames.append(os.path.splitext(file)[0])  # Remove the .mp4 extension
    return filenames
print(get_filenames())

## INICIALIAZACAO E USO DOS LANDMARKERS
for filename in get_filenames():
    with PoseLandmarker.create_from_options(options_pose) as pose_landmarker:
        with HandLandmarker.create_from_options(options_hand) as hand_landmarker:
            video_path = f"./DB/{filename}.mp4"
            output_path = f"./LANDMARKS-POINTS/{filename}-landmarks.pkl"  # Path to save the output file

            print(f"Iniciando processamento do video {filename}")
            if os.path.exists(output_path):
                print(f"O arquivo '{filename}' já existe. O video não será analisado.")
                continue
            # The landmarker is initialized. Use it here.

            print(video_path)
            cap = cv2.VideoCapture(video_path)

            fps = cap.get(cv2.CAP_PROP_FPS)
            #print(fps)

            pose_landmarks = []
            hand_landmarks = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
            
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # Current frame number
                frame_timestamp_ms = int((frame_index / fps) * 1000)
                
                # Perform pose landmarking on the provided single image.
                # The pose landmarker must be created with the video mode.
                pose_landmarker_result = pose_landmarker.detect_for_video(mp_image, frame_timestamp_ms)      
                hand_landmarker_result = hand_landmarker.detect_for_video(mp_image, frame_timestamp_ms)

                pose_landmarks.append(pose_landmarker_result)
                hand_landmarks.append(hand_landmarker_result)

            # Release the VideoWriter
            cap.release()

            landmarks_results = {"pose_landmarks": pose_landmarks, "hand_landmarks": hand_landmarks}
            if not os.path.exists(output_path):
                with open(output_path, "wb") as f:
                    pickle.dump(landmarks_results, f)
                    print(f"Objeto salvo em {output_path}.")
            else:
                print(f"O arquivo '{output_path}' já existe. O objeto não foi salvo.")