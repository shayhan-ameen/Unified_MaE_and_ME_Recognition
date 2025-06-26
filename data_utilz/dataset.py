# from landmarks import detect_landmarks
from data_utilz.landmarks import detect_landmarks
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
import cv2
from scipy import signal
from math import ceil
import os
import pickle


def find_frame_path(catego, image_root, subject, folder, frame_name):
    if catego == "SAMM":
        frame_path = f"{image_root}/{subject}/{folder}/{subject}_{frame_name:05}.jpg"
    elif catego == "CAS(ME)^2":
        frame_path = f"{image_root}/s{subject}/{folder}/img{frame_name}.jpg"
    else:
        frame_path = f"{image_root}/sub{subject}/{folder}/img{frame_name}.jpg"

    return frame_path


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

def kirsch_mask(img):
    # kirsch_mask kernel
    na= np.array([[-3,-3,5],[-3,0,5],[-3,-3,5]])
    nwa= np.array([[-3,5,5],[-3,0,5],[-3,-3,-3]])
    wa= np.array([[5,5,5],[-3,0,-3],[-3,-3,-3]])
    swa= np.array([[5,5,-3],[5,0,-3],[-3,-3,-3]])
    sa= np.array([[5,-3,-3],[5,0,-3],[5,-3,-3]])
    sea= np.array([[-3,-3,-3],[5,0,-3],[5,5,-3]])
    ea= np.array([[-3,-3,-3],[-3,0,-3],[5,5,5]])
    nea= np.array([[-3,-3,-3],[-3,0,5],[-3,5,5]])
    kernels=  [na, nwa, wa, swa, sa, sea, ea, nea]
    output= np.stack([signal.correlate2d(img,kernel,mode='same') for kernel in kernels], axis=2)
    # return 8*output.argmax(axis=2) + output.argmin(axis=2)
    return output


class MMDataset(Dataset):
    def __init__(self, data_info: pd.DataFrame, label_mapping: dict, image_root: str, catego: str, device: torch.device, train: bool, num_classes=5, num_frames=3):

        self.image_root = image_root
        self.data_info = data_info
        self.label_mapping = label_mapping
        self.catego = catego
        self.train = train
        self.device = device
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.info = pickle.load(open(f'data/processed_dataset/{self.catego}_processed.pkl', 'rb'))

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx: int):
        subject = self.data_info.loc[idx, "Subject"]
        onset_name = self.data_info.loc[idx, "OnsetFrame"]
        folder = self.data_info.loc[idx, "Filename"]
        # print(str(subject)+'_'+folder+'_'+str(onset_name))
        feature_appearances, stldn_sequences = self.info[str(subject)+'_'+folder+'_'+str(onset_name)]

        emotion_categories = "Estimated Emotion " + str(self.num_classes)
        labels = self.label_mapping[self.data_info.loc[idx, emotion_categories]]
        # if label !=labels:
        #     print(f'Label Problem--------Orioginal: {label} | Found {labels}>')

        # return feature_coordinates, feature_appearances, frame_sequences, tldn_sequences, apex_frame, tldn_single, labels
        return feature_appearances, stldn_sequences, labels

