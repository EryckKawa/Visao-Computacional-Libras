import mediapipe as mp 
from mediapipe.tasks import python 
from mediapipe.tasks.python import vision
import cv2
import matplotlib.pyplot as plt

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

import os

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

MY_POSE_CONNECTIONS = frozenset([(x,y) for (x,y) in solutions.pose.POSE_CONNECTIONS if x < 17 and y < 17])

model_pose_path = './tasks/pose_landmarker_heavy.task'
model_hand_path = './tasks/hand_landmarker.task'

####################### FUNÇÃO DE DESENHO #############################

def draw_landmarks_on_image(rgb_image, hand_detection_result, pose_detection_result):
  # landmarks da mão
  hand_landmarks_list = hand_detection_result.hand_landmarks
  handedness_list = hand_detection_result.handedness
  
  # landmarks de pose
  pose_landmarks_list = pose_detection_result.pose_landmarks
  
  annotated_image = np.copy(rgb_image)

  # Percorra através das mãos detectadas
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for (i, landmark) in enumerate(pose_landmarks) #if i < 17
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS, #MY_POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

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

i=1

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
with PoseLandmarker.create_from_options(options_pose) as pose_landmarker:
    with HandLandmarker.create_from_options(options_hand) as hand_landmarker:
        for filename in get_filenames():
            video_path = f"./DB/{filename}.mp4"
            output_path = f"./ANNOTATED-DB/{filename}-landmarks.mp4"  # Path to save the output video

            print(f"Iniciando processamento do video {filename}")
            if os.path.exists(output_path):
                print(f"O arquivo '{filename}' já existe. O video não será analisado.")
                continue
            # The landmarker is initialized. Use it here.

            print(video_path)
            cap = cv2.VideoCapture(video_path)

            fps = cap.get(cv2.CAP_PROP_FPS)

            annotated_frames = []

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

                landmarks_results = [pose_landmarker_result, hand_landmarker_result]

                annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), hand_landmarker_result, pose_landmarker_result)

                annotated_frames.append(annotated_image)

            frame_height, frame_width, _ = annotated_frames[0].shape  # Height and width from the first frame

            # Define the codec and create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

            # Write each frame to the video
            for frame in annotated_frames:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV

            # Release the VideoWriter
            out.release()

            print(f"Video saved as {output_path}")