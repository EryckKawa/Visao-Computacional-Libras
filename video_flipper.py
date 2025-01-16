# FLIP HORIZONTAL DOS VIDEOS PARA AUMENTAR A BASE DE DADOS

import cv2
import os

def get_filenames():
    # Path to the folder containing the files
    folder_path = "./DB"

    # List to store filenames without .mp4 extension
    filenames = []

    # Get all files in the folder
    for file in os.listdir(folder_path):
        if file.endswith(".mp4") and not file.endswith("-mirrored.mp4"):
            filenames.append(os.path.splitext(file)[0])  # Remove the .mp4 extension
    return filenames

for filename in get_filenames():
    # Input and output video paths
    input_video = f"./DB/{filename}.mp4"
    output_video = f"./DB/{filename}-mirrored.mp4"

    if os.path.exists(output_video):
        print(f"O arquivo '{filename}' já existe. O video não flipado.")
        continue
    
    # Open the input video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {input_video}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror the frame horizontally
        mirrored_frame = cv2.flip(frame, 1)  # 1 for horizontal flip

        # Write the mirrored frame to the output video
        out.write(mirrored_frame)

    # Release resources
    cap.release()
    out.release()

    print(f"Mirrored video saved as {output_video}")
