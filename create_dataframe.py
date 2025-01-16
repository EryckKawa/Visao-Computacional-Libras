import os
import pickle
import pandas as pd
import numpy as np

def get_filenames():
    # Path to the folder containing the files
    folder_path = "./LANDMARKS-POINTS"

    return os.listdir(folder_path)

#print("\n".join(get_filenames()))

def ponto_interpolado(ponto_a, ponto_b, razao):
    x1, y1 = ponto_a
    x2, y2 = ponto_b

    # Calcular o ponto interpolado
    x = x1 + razao * (x2 - x1)
    y = y1 + razao * (y2 - y1)

    return (x, y)

def get_estimated_left_hand_points(important_pose_landmarks):
    ponto_1 = ponto_interpolado(important_pose_landmarks[19], important_pose_landmarks[17], 1/3)
    ponto_2 = ponto_interpolado(important_pose_landmarks[19], important_pose_landmarks[17], 2/3)

    return [important_pose_landmarks[15], # base da mao
            important_pose_landmarks[21], # polegar
            important_pose_landmarks[21],
            important_pose_landmarks[21],
            important_pose_landmarks[21],
            important_pose_landmarks[19], # indicador
            important_pose_landmarks[19],
            important_pose_landmarks[19],
            important_pose_landmarks[19],
            ponto_1, # medio
            ponto_1,
            ponto_1,
            ponto_1,
            ponto_2, # anelar
            ponto_2,
            ponto_2,
            ponto_2,
            important_pose_landmarks[17], # minimo
            important_pose_landmarks[17],
            important_pose_landmarks[17],
            important_pose_landmarks[17],
            ]

def get_estimated_right_hand_points(important_pose_landmarks):
    ponto_1 = ponto_interpolado(important_pose_landmarks[20], important_pose_landmarks[18], 1/3)
    ponto_2 = ponto_interpolado(important_pose_landmarks[20], important_pose_landmarks[18], 2/3)

    return [important_pose_landmarks[16], # base da mao
            important_pose_landmarks[22], # polegar
            important_pose_landmarks[22],
            important_pose_landmarks[22],
            important_pose_landmarks[22],
            important_pose_landmarks[20], # indicador
            important_pose_landmarks[20],
            important_pose_landmarks[20],
            important_pose_landmarks[20],
            ponto_1, # medio
            ponto_1,
            ponto_1,
            ponto_1,
            ponto_2, # anelar
            ponto_2,
            ponto_2,
            ponto_2,
            important_pose_landmarks[18], # minimo
            important_pose_landmarks[18],
            important_pose_landmarks[18],
            important_pose_landmarks[18],
            ]

def process_landmarks(selected_frames, hand_list, pose_list):
    #IMPLEMENTAR
    processed_landmarks = []
    for frame in selected_frames:
        processed_frame_landmarks = []
        #a = (x for x in pose_list[frame][0][:23])
        important_pose_landmarks = [(landmark.x, landmark.y) for landmark in pose_list[frame][0][:23]]
        
        try:
            important_hand_one_landmarks = [(landmark.x, landmark.y) for landmark in hand_list[frame][0]]
        except:
            important_hand_one_landmarks = get_estimated_left_hand_points(important_pose_landmarks)
        try:
            important_hand_two_landmarks = [(landmark.x, landmark.y) for landmark in hand_list[frame][1]]
        except:
            important_hand_two_landmarks = get_estimated_right_hand_points(important_pose_landmarks)
        #print(important_hand_one_landmarks)

        processed_frame_landmarks.extend(important_pose_landmarks)
        processed_frame_landmarks.extend(important_hand_one_landmarks)
        processed_frame_landmarks.extend(important_hand_two_landmarks)

        processed_landmarks.append(processed_frame_landmarks)
        #break
    return processed_landmarks
    #pass

def relative_position_list(landmarks_list):
    p11 = landmarks_list[11]
    p12 = landmarks_list[12]
    p_medio = (
        (p12[0] + p11[0]) / 2,
        (p12[1] + p11[1]) / 2
    )
    posicoes_relativas = [
        (p[0] - p_medio[0], p[1] - p_medio[1])
        for p in landmarks_list
    ]
    return posicoes_relativas

def normalize_landmarks(landmarks_list):
    relative_landmarks = relative_position_list(landmarks_list)
    
    x_values = [p[0] for p in relative_landmarks]
    y_values = [p[1] for p in relative_landmarks]

    # Find min and max for x and y
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)

    normalized_points = [
        ((x - x_min) / (x_max - x_min), (y - y_min) / (y_max - y_min))
        for x, y in relative_landmarks
    ]

    return normalized_points

frames = []

def save_video_tensor(video_tensor, output_dir, filename):
    """
    Save a video tensor to a file in .npy format.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output file path
    output_path = os.path.join(output_dir, f"{filename}.npy")
    
    # Save the tensor
    np.save(output_path, video_tensor)

for filename in get_filenames():
    with open(f"./LANDMARKS-POINTS/{filename}", "rb") as f:
        landmarks_results = pickle.load(f)
        # landmarks_results -> dicionário com 2 entradas: 'hand_landmarks' e 'pose_landmarks'
        # landmarks_results['hand_landmarks'] -> (lista de hand landmarks do video inteiro)
        # landmarks_results['pose_landmarks'] -> (lista de pose landmarks do video inteiro)
        # landmarks_results['pose_landmarks'][i] -> (acessa os NormalizedLandmarks do frame i do video)
        # landmarks_results['hand_landmarks'][i].hand_landmarks -> (acessa os landmarks do frame i do video) - igual à quantidade mãos no video
        # landmarks_results['hand_landmarks'][i].handedness -> (acessa a lista dos lados da mao dos landmarks do frame i do video) - igual à quantidade mãos no video
        # landmarks_results['hand_landmarks'][i].hand_landmarks[m][j] -> (acessa o landmark j da mao m do frame i) - (coordenada (x,y,z))

        # landmarks_results['pose_landmarks'][i].pose_landmarks -> (acessa os landmarks do frame i do video) - igual à quantidade de poses no video
        # landmarks_results['hand_landmarks'][i].hand_landmarks[p][j] -> (acessa o landmark j da pose p do frame i) - (coordenada (x,y,z))

        hand_list = [sorted(zip(x.hand_landmarks, x.handedness), key=lambda pair: pair[1][0].category_name) for x in landmarks_results['hand_landmarks']]
        hand_list = [[landmark for landmark, _ in hands] for hands in hand_list]
        pose_list = [x.pose_landmarks for x in landmarks_results['pose_landmarks']]

        #debugging
        #print(len(pose_list[0][0]))
        #print(len(pose_list[10][0][:23]))

        n_frames = len(landmarks_results['hand_landmarks'])
        
        frames.append(n_frames)
        #print(n_frames)

        markations = 41

        bound = n_frames//markations
        
        selected_frames = [bound * (x+1) for x in range(markations - 1)]

        #print(selected_frames)

        selected_landmarks = process_landmarks(selected_frames, hand_list, pose_list)

        #print(len(selected_landmarks[0]))

        normalized_landmarks = [normalize_landmarks(frame_landmarks) for frame_landmarks in selected_landmarks]

        landmarks = [pd.DataFrame(frame_landmarks) for frame_landmarks in normalized_landmarks]

        landmarks_as_arrays = [df.values for df in landmarks]

        video_tensor = np.array(landmarks_as_arrays)

        print(video_tensor.shape)

        save_video_tensor(video_tensor, "./VIDEO-TENSORS", filename.split('.')[0])
        #print(len(normalized_landmarks))
        #print(len(landmarks_results['hand_landmarks'][51].hand_landmarks))
#print(sum(frames)/len(frames))