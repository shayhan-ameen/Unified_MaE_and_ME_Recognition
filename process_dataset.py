import argparse
from data_utilz.landmarks import detect_landmarks
from data_utilz.kirsch_mask import kirsch_mask_model
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
import cv2
from scipy import signal
from math import ceil
import os
from read_file import read_csv
from data_utilz.dataloader import train_val_test_split
import pickle
import math

def landmark_motion_based_keyframe_selection(path, start_frame, end_frame, k):
    """
    Selects keyframes from a sequence of images based on the motion of facial landmarks.

    This function calculates the motion between consecutive frames by measuring the Euclidean distance between corresponding facial landmarks.
    It then identifies the top `k` frames with the highest total landmark motion, which are considered keyframes.

    Parameters:
    path (str): The directory path where the images are stored. Images should be named as 'img{frame_number}.jpg'.
    start_frame (int): The starting frame number in the sequence.
    end_frame (int): The ending frame number in the sequence.
    k (int): The number of keyframes to select based on the highest landmark motion.

    Returns:
    np.ndarray: An array of indices representing the frames with the highest landmark motion, corresponding to the selected keyframes.

    """
    total_movement = []
    for i in range(start_frame,end_frame):
        frame = cv2.imread(f'{path}img{i}.jpg', 0)
        landmarks = detect_landmarks(frame)
        frame_motion = 0
        if i!=start_frame:
            for previous_point, point in zip(landmarks,previous_landmarks):
                frame_motion+= math.dist(previous_point, point)
                total_movement.append(frame_motion)
        previous_landmarks = landmarks

    # Find the top k points
    top_k_indices = np.argsort(total_movement)[-k:]
    return top_k_indices

def stldn_onset_intermediate_apx(catego,  image_root, subject, folder, previous, current, next):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kirsch_mask_detector = kirsch_mask_model()
    kirsch_mask_detector = kirsch_mask_detector.to(device)

    frames = [previous, current, next]
    ldn = []

    for frame_name in frames:
        frame_path = find_frame_path(catego, image_root, subject, folder, frame_name)
        frame = read_frame(frame_path, catego)
        if catego == "SAMM":
            frame = center_crop(frame, (420, 420))
        points = detect_landmarks(frame)  # detect 51 (68-17) landmarks
        frame = torch.Tensor(frame)
        frame = frame.to(device)
        ldn.append(kirsch_mask_detector(frame))

    output = torch.dstack([ldn[0], ldn[1], ldn[2]])
    stldn = ((32 * output.argmax(axis=2)) + output.argmin(axis=2)) / 758.0  # 10111 10110

    patches = []
    for point in points:
        start_x, end_x, start_y, end_y = get_patches(point)
        patches.append(stldn[start_y:end_y, start_x:end_x].cpu().numpy())

    feature_single = torch.FloatTensor(patches)  # 51 * 7 * 7

    return feature_single

def frame_num_selection(dataset_name):
    """
    A function to find the minimum frame between onset and apex frame

    Parameters
    ----------
    dataset_name : Ex CSA(ME)^2

    Returns
    -------

    """

    __minFrame = {
        'CASME I': 10,
        'CASME II': 10,
        'CAS(ME)^2': 10,
        'CAS(ME)^3': 10,
        'SAM': 10,
    }
    return __minFrame[dataset_name]

def find_frame_path(catego, image_root, subject, folder, frame_name):
    if catego == "SAMM":
        frame_path = f"{image_root}/{subject}/{folder}/{subject}_{frame_name:05}.jpg"
    elif catego == "CAS(ME)^2":
        frame_path = f"{image_root}/s{subject}/{folder}/img{frame_name}.jpg"
    elif catego == "CAS(ME)^3":
        frame_path = f"{image_root}/{subject}/{folder}/color/{int(frame_name)}.jpg"
        # print(frame_path)
    elif catego == "MMEW":
        if int(subject) < 10:
            frame_path = f"{image_root}/S0{subject}/{folder}/{frame_name}.jpg"
    else:
        frame_path = f"{image_root}/sub{subject}/{folder}/img{frame_name}.jpg"

    return frame_path

def read_frame(frame_path, catego, color_flag=0, h=300, w=320):
    # color_flag 0 = gray, 1 = color
    if catego == 'CAS(ME)^3':
        new_frame = cv2.imread(frame_path, color_flag)
        new_frame = cv2.resize(new_frame, (w, h))
    else:
        new_frame = cv2.imread(frame_path, color_flag)
    return new_frame

def center_crop(img: np.array, crop_size) -> np.array:
    """Returns center cropped image

    Parameters
    ----------
    img : [type]
        Image to do center crop
    crop_size : Union[tuple, int]
        Crop size of the image

    Returns
    -------
    np.array
        Image after being center crop
    """
    width, height = img.shape[1], img.shape[0]

    # Height and width of the image
    mid_x, mid_y = int(width / 2), int(height / 2)

    if isinstance(crop_size, tuple):
        crop_width, crop_hight = int(crop_size[0] / 2), int(crop_size[1] / 2)
    else:
        crop_width, crop_hight = int(crop_size / 2), int(crop_size / 2)
    crop_img = img[mid_y - crop_hight:mid_y + crop_hight, mid_x - crop_width:mid_x + crop_width]

    return crop_img

def get_patches(point: tuple):
    start_x = point[0] - 3
    end_x = point[0] + 4

    start_y = point[1] - 3
    end_y = point[1] + 4

    return start_x, end_x, start_y, end_y

def dataset_process(csv_path, image_root, num_classes, catego, num_frames=3):
    """

    Parameters
    ----------
    csv_path
    image_root
    num_classes
    catego
    num_frames

    Returns
    -------
    frames=[557, 562, 567, 572, 573]
    feature_coordinates.shape=torch.Size([5, 51, 2])
    feature_appearances.shape=torch.Size([153, 7, 7])
    frame_sequences.shape=torch.Size([5, 256, 256])
    stldn_sequences.shape=torch.Size([3, 480, 640])
    labels=0

    """
    num_frames = frame_num_selection(catego)
    data, label_mapping = read_csv(csv_path, num_classes)
    emotion_categories = "Estimated Emotion " + str(num_classes)
    info = {}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kirsch_mask_detector = kirsch_mask_model()
    kirsch_mask_detector = kirsch_mask_detector.to(device)

    # for idx in range(790, 860):
    for idx in data.index:

        labels = label_mapping[data.loc[idx, emotion_categories]]
        subject = data.loc[idx, "Subject"]
        onset_name = data.loc[idx, "OnsetFrame"]
        apex_name = data.loc[idx, "ApexFrame"]
        folder = data.loc[idx, "Filename"]
        print(f'idx={idx+2}, {subject=}, {folder=}, {onset_name=}')
        # print(f'--------{type(onset_name)=}, {onset_name=}')

        stldn_single = stldn_onset_intermediate_apx(catego,  image_root, subject, folder, onset_name, int((int(onset_name)+int(apex_name))/2.0), apex_name)

        frame_sequences = []
        stldn_sequences = []
        feature_coordinates = []
        feature_appearances = []
        frames = []

        # frames selection NEW
        diff = (int(apex_name) - int(onset_name)) / num_frames
        for i in range(num_frames - 1):
            frames.append((int(onset_name) + ceil(diff * i)))
        frames.append(apex_name)
        frames.append(int(apex_name) + 1)

        # frames selection OLD
        # distance = int(apex_name) - int(onset_name)
        # frames.append(onset_name)
        # next_frame = onset_name
        # for i in range(num_frames - 1):
        #     next_frame = next_frame + ceil(distance / num_frames)
        #     frames.append(next_frame)
        # frames.append(apex_name)
        # frames.append(int(apex_name)+1)

        frame_path = find_frame_path(catego, image_root, subject, folder, onset_name)
        frame_previous = read_frame(frame_path, catego)
        frame_path = find_frame_path(catego, image_root, subject, folder, frames[1])
        frame_current = read_frame(frame_path, catego)

        if catego == "SAMM":
            frame_previous = center_crop(frame_previous, (420, 420))
            frame_current = center_crop(frame_current, (420, 420))

        feature_coordinates.append(detect_landmarks(frame_previous))
        feature_coordinates.append(detect_landmarks(frame_current))


        resized_fame = cv2.resize(frame_previous, dsize=(256, 256), interpolation=cv2.INTER_AREA) / 255.0
        frame_sequences.append(resized_fame)
        resized_fame = cv2.resize(frame_current, dsize=(256, 256), interpolation=cv2.INTER_AREA) / 255.0
        frame_sequences.append(resized_fame)

        frame_current_face = frame_current
        frame_previous = torch.Tensor(frame_previous)
        frame_current = torch.Tensor(frame_current)
        frame_previous = frame_previous.to(device)
        frame_current = frame_current.to(device)
        ldn_previous = kirsch_mask_detector(frame_previous)
        ldn_current = kirsch_mask_detector(frame_current)

        # print(f'{idx=} {frames}')


        for frame_name in frames[2:]:
            # frames[0] - previous frame, frames[1] - current frame
        # for frame_name in range(strat_name+1, apex_name + 2):
            # Create the path for the frame and Read the frame

            frame_path = find_frame_path(catego, image_root, subject, folder, frame_name)
            if os.path.isfile(frame_path) is False:
                frame_path = find_frame_path(catego, image_root, subject, folder, frame_name - 1)
            frame_next = read_frame(frame_path, catego)


            if catego == "SAMM":
                frame_next = center_crop(frame_next, (420, 420))

            resized_fame = cv2.resize(frame_next, dsize=(256, 256), interpolation=cv2.INTER_AREA) / 255.0
            frame_sequences.append(resized_fame)

            # Preprocessing of the image

            # Landmarks detection
            points = detect_landmarks(frame_current_face) # detect 51 (68-17) landmarks
            frame_current_face = frame_next

            # freature_appearance - VLDN -> cnn -> avg pooling R (Channel×Patch×Time)
            frame_next = torch.Tensor(frame_next)
            frame_next = frame_next.to(device)
            ldn_next = kirsch_mask_detector(frame_next)
            output = torch.dstack([ldn_previous, ldn_current, ldn_next])
            # vldn = 8*(output.argmax(axis=2)%8) + (output.argmin(axix=2)%8)
            stldn = ((32 * output.argmax(axis=2)) + output.argmin(axis=2)) / 986 # 758.0  # 10111 10110
            stldn_sequences.append(stldn)

            patches = []
            for point in points:
                start_x, end_x, start_y, end_y = get_patches(point)
                patches.append(stldn[start_y:end_y, start_x:end_x].cpu().numpy())

            patches = torch.FloatTensor(patches) # 51 * 7 * 7

            ldn_previous = ldn_current
            ldn_current = ldn_next

            feature_coordinates.append(points) # 3*51*2 -> T*V*C
            feature_appearances.append(patches)  # 3*51*7*7 -> T*V*C


        # print(points)
        feature_appearances = torch.stack(feature_appearances, dim=0)
        T, V, H, W = feature_appearances.size() # 3*51*7*7 -> T*V*C
        # f = f.permute(1, 0, 5, 2, 3, 4).contiguous().view(I, N*M, C, T, V)
        feature_appearances = feature_appearances.view(T * V, H, W) # (3*51)*7*7 -> 153*7*7 -> T*V*C
        # f = torch.unsqueeze(f, dim=0)

        feature_coordinates = np.array(feature_coordinates)
        feature_coordinates = torch.FloatTensor(feature_coordinates)
        T, V, W = feature_coordinates.size()  # 3*51*2 -> T*V*C
        feature_coordinates = feature_coordinates.view(T * V,  W)  # (3*51)*2 -> 153*2-> T*V*C

        frame_sequences = torch.FloatTensor(frame_sequences) / 255.0

        stldn_sequences = torch.stack(stldn_sequences, dim=0)

        # for Resnet
        frame_path = find_frame_path(catego, image_root, subject, folder, apex_name)
        apex_frame = read_frame(frame_path, catego, color_flag=1)
        if catego == "SAMM":
            frame_next = center_crop(apex_frame, (420, 420))
        apex_frame = cv2.resize(apex_frame, dsize=(224, 224), interpolation=cv2.INTER_AREA) / 255.0
        apex_frame = torch.FloatTensor(apex_frame)
        apex_frame = apex_frame.permute(2,0,1)


        # info[str(subject)+'_'+folder+'_'+str(onset_name)] = [feature_coordinates, feature_appearances, frame_sequences, stldn_sequences.cpu(), apex_frame, stldn_single, labels]
        info[str(subject) + '_' + folder + '_' + str(onset_name)] = [feature_appearances, stldn_sequences.cpu()]

    pickle.dump(info, open(f'data/processed_dataset/{catego}_processed.pkl', 'wb'))
    print(f'--------- Done! Dataset size: {len(info)} ---------')

if __name__ == "__main__":
    # Argument parse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True, help="Path for the csv file for training data")
    parser.add_argument("--image_root", type=str, required=True, help="Root for the training images")
    parser.add_argument("--catego", type=str, required=True, help="SAMM or CASME^2 dataset")
    parser.add_argument("--num_classes", type=str, default="Folder", help="Classes to be trained")
    args = parser.parse_args()

    # csv_path= "../data/csv/casmesquare.csv"
    # num_classes="Folder"
    # image_root= "D:/Datasets/Image/CAS(ME)^2/casme/selectedpic"
    # catego = 'CAS(ME)^2'

    dataset_process(args.csv_path, args.image_root, args.num_classes, args.catego, num_frames=3)
