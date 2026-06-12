"""
Dataset preprocessing script for unified macro-expression and micro-expression recognition.

This script converts raw extracted video frames into two model-ready inputs:

    1. Landmark-centered STLDN patches
       Used by the Facial Graph Stream.

       Shape:
           feature_appearances: [(T * V), 7, 7]

    2. Full STLDN image sequence
       Used by the Visual Stream.

       Shape:
           stldn_sequences: [T, H, W]

where:
    T = number of generated STLDN maps
    V = number of selected facial landmarks, usually 51
    H = frame height
    W = frame width

Main pipeline:
    1. Read dataset metadata from a CSV file.
    2. Select frames between onset and apex.
    3. Process consecutive frame triplets:
           previous frame, current frame, next frame
    4. Apply Kirsch masks to obtain directional edge responses.
    5. Generate STLDN feature maps.
    6. Detect 51 inner facial landmarks.
    7. Crop 7 x 7 STLDN patches around each landmark.
    8. Save processed features as a pickle file.

Saved file:
    data/processed_dataset/{catego}_processed.pkl

Each saved sample:
    info[sample_key] = [
        feature_appearances,   # [(T * 51), 7, 7]
        stldn_sequences        # [T, H, W]
    ]
"""

import argparse
import math
import os
import pickle
from math import ceil
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from data_utilz.kirsch_mask import kirsch_mask_model
from data_utilz.landmarks import detect_landmarks
from read_file import read_csv

STLDN_MAX_VALUE = 986.0
PATCH_RADIUS = 3
PATCH_SIZE = 2 * PATCH_RADIUS + 1


def landmark_motion_based_keyframe_selection(
    path: str,
    start_frame: int,
    end_frame: int,
    k: int,
) -> List[int]:
    """
    Select keyframes based on facial landmark motion.

    The motion score of each frame is computed as the sum of Euclidean
    distances between corresponding landmarks in two consecutive frames.

    Args:
        path:
            Directory containing frame images.

            Expected naming:
                img{frame_number}.jpg

        start_frame:
            First frame number.

        end_frame:
            Last frame number.

        k:
            Number of keyframes to select.

    Returns:
        selected_frames:
            Sorted list of selected frame numbers.

    Important:
        The original version stored partial motion scores inside the landmark
        loop. This fixed version stores exactly one motion score per frame.
    """

    frame_numbers = []
    total_movement = []
    previous_landmarks = None

    for frame_idx in range(start_frame, end_frame + 1):
        frame_path = os.path.join(path, f"img{frame_idx}.jpg")
        frame = cv2.imread(frame_path, 0)

        if frame is None:
            continue

        landmarks = detect_landmarks(frame)

        if previous_landmarks is not None:
            frame_motion = 0.0

            for previous_point, current_point in zip(previous_landmarks, landmarks):
                frame_motion += math.dist(previous_point, current_point)

            frame_numbers.append(frame_idx)
            total_movement.append(frame_motion)

        previous_landmarks = landmarks

    if len(total_movement) == 0:
        return []

    k = min(k, len(total_movement))

    top_k_indices = np.argsort(total_movement)[-k:]
    selected_frames = [frame_numbers[i] for i in top_k_indices]
    selected_frames = sorted(selected_frames)

    return selected_frames


def frame_num_selection(dataset_name: str) -> int:
    """
    Return the number of selected frames for each dataset.

    Args:
        dataset_name:
            Dataset name.

    Returns:
        Number of selected frames.

    Notes:
        This implementation uses 10 frames for all listed datasets.
    """

    num_frames_by_dataset = {
        "CASME I": 10,
        "CASME II": 10,
        "CAS(ME)^2": 10,
        "CAS(ME)^3": 10,
        "SAM": 10,
        "SAMM": 10,
        "MMEW": 10,
    }

    if dataset_name not in num_frames_by_dataset:
        raise ValueError(
            f"Unknown dataset name: {dataset_name}. "
            f"Supported datasets: {list(num_frames_by_dataset.keys())}"
        )

    return num_frames_by_dataset[dataset_name]


def find_frame_path(catego, image_root, subject, folder, frame_name):
    """
    Build the frame path for different dataset folder structures.

    Args:
        catego:
            Dataset name.

        image_root:
            Root directory of extracted frames.

        subject:
            Subject ID.

        folder:
            Sequence folder name.

        frame_name:
            Frame number.

    Returns:
        frame_path:
            Full path to the frame image.
    """

    if catego == "SAMM":
        frame_path = (
            f"{image_root}/{subject}/{folder}/{subject}_{int(frame_name):05}.jpg"
        )

    elif catego == "CAS(ME)^2":
        frame_path = f"{image_root}/s{subject}/{folder}/img{frame_name}.jpg"

    elif catego == "CAS(ME)^3":
        frame_path = f"{image_root}/{subject}/{folder}/color/{int(frame_name)}.jpg"

    elif catego == "MMEW":
        if int(subject) < 10:
            frame_path = f"{image_root}/S0{subject}/{folder}/{frame_name}.jpg"
        else:
            frame_path = f"{image_root}/S{subject}/{folder}/{frame_name}.jpg"

    else:
        frame_path = f"{image_root}/sub{subject}/{folder}/img{frame_name}.jpg"

    return frame_path


def read_frame(frame_path, catego, color_flag=0, h=300, w=320):
    """
    Read a frame from disk.

    Args:
        frame_path:
            Full path to the frame.

        catego:
            Dataset name.

        color_flag:
            OpenCV reading flag.

            0 = grayscale
            1 = color

        h:
            Resize height for CAS(ME)^3.

        w:
            Resize width for CAS(ME)^3.

    Returns:
        frame:
            Loaded frame as a NumPy array.

    Raises:
        FileNotFoundError:
            If OpenCV cannot read the frame.
    """

    frame = cv2.imread(frame_path, color_flag)

    if frame is None:
        raise FileNotFoundError(f"Could not read frame: {frame_path}")

    if catego == "CAS(ME)^3":
        frame = cv2.resize(frame, (w, h))

    return frame


def center_crop(img: np.ndarray, crop_size) -> np.ndarray:
    """
    Center-crop an image.

    Args:
        img:
            Input image.

            Shape:
                [H, W] or [H, W, C]

        crop_size:
            Crop size.

            If tuple:
                crop_size = (width, height)

            If int:
                crop_size is used for both width and height.

    Returns:
        crop_img:
            Center-cropped image.
    """

    height, width = img.shape[:2]
    mid_x = width // 2
    mid_y = height // 2

    if isinstance(crop_size, tuple):
        crop_width = crop_size[0] // 2
        crop_height = crop_size[1] // 2
    else:
        crop_width = crop_size // 2
        crop_height = crop_size // 2

    crop_img = img[
        mid_y - crop_height : mid_y + crop_height,
        mid_x - crop_width : mid_x + crop_width,
    ]

    return crop_img


def get_patches(point: Tuple[int, int]):
    """
    Return 7 x 7 patch boundaries around a landmark point.

    Args:
        point:
            Landmark point as (x, y).

    Returns:
        start_x, end_x, start_y, end_y:
            Patch boundaries.

    Notes:
        This function only returns raw boundaries.
        For boundary-safe patch extraction, use crop_landmark_patch().
    """

    x = int(round(point[0]))
    y = int(round(point[1]))

    start_x = x - PATCH_RADIUS
    end_x = x + PATCH_RADIUS + 1

    start_y = y - PATCH_RADIUS
    end_y = y + PATCH_RADIUS + 1

    return start_x, end_x, start_y, end_y


def crop_landmark_patch(stldn: torch.Tensor, point: Tuple[int, int]) -> torch.Tensor:
    """
    Crop a boundary-safe 7 x 7 STLDN patch around a landmark.

    This function pads the STLDN map before cropping, so it always returns a
    fixed-size patch even when the landmark is near the image boundary.

    Args:
        stldn:
            STLDN map with shape [H, W].

        point:
            Landmark coordinate as (x, y).

    Returns:
        patch:
            STLDN patch with shape [7, 7].
    """

    if stldn.dim() != 2:
        raise ValueError(
            f"Expected STLDN map with shape [H, W], got {tuple(stldn.shape)}"
        )

    x = int(round(point[0]))
    y = int(round(point[1]))

    padded = (
        F.pad(
            stldn.unsqueeze(0).unsqueeze(0),
            pad=(PATCH_RADIUS, PATCH_RADIUS, PATCH_RADIUS, PATCH_RADIUS),
            mode="constant",
            value=0.0,
        )
        .squeeze(0)
        .squeeze(0)
    )

    x = x + PATCH_RADIUS
    y = y + PATCH_RADIUS

    patch = padded[
        y - PATCH_RADIUS : y + PATCH_RADIUS + 1,
        x - PATCH_RADIUS : x + PATCH_RADIUS + 1,
    ]

    if patch.shape != (PATCH_SIZE, PATCH_SIZE):
        raise RuntimeError(
            f"Invalid patch shape {tuple(patch.shape)} for landmark {point}. "
            f"Expected {(PATCH_SIZE, PATCH_SIZE)}."
        )

    return patch


def generate_stldn_from_ldn(ldn_previous, ldn_current, ldn_next):
    """
    Generate an STLDN feature map from three Kirsch response maps.

    Args:
        ldn_previous:
            Kirsch response of previous frame.

        ldn_current:
            Kirsch response of current frame.

        ldn_next:
            Kirsch response of next frame.

    Returns:
        stldn:
            Normalized STLDN map with shape [H, W].

    Formula:
        STLDN = 32 * argmax(response) + argmin(response)

    Normalization:
        The manuscript states that STLDN values lie in [0, 986].
        Therefore, this implementation normalizes by 986.0.
    """

    output = torch.dstack([ldn_previous, ldn_current, ldn_next])
    stldn = ((32 * output.argmax(dim=2)) + output.argmin(dim=2)) / STLDN_MAX_VALUE

    return stldn


def stldn_onset_intermediate_apx(
    catego,
    image_root,
    subject,
    folder,
    previous,
    current,
    next,
):
    """
    Generate landmark-centered STLDN patches using onset, intermediate, and apex frames.

    Args:
        catego:
            Dataset name.

        image_root:
            Root image directory.

        subject:
            Subject ID.

        folder:
            Sequence folder.

        previous:
            Previous frame number, usually onset.

        current:
            Current frame number, usually middle frame.

        next:
            Next frame number, usually apex.

    Returns:
        feature_single:
            Tensor with shape [51, 7, 7].

    Note:
        This helper is kept for compatibility. In the current dataset_process()
        function, the returned value is not saved.
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kirsch_mask_detector = kirsch_mask_model().to(device)

    frames = [previous, current, next]
    ldn = []
    points = None

    for frame_name in frames:
        frame_path = find_frame_path(catego, image_root, subject, folder, frame_name)
        frame = read_frame(frame_path, catego)

        if catego == "SAMM":
            frame = center_crop(frame, (420, 420))

        points = detect_landmarks(frame)

        frame_tensor = torch.tensor(frame, dtype=torch.float32, device=device)
        ldn.append(kirsch_mask_detector(frame_tensor))

    stldn = generate_stldn_from_ldn(ldn[0], ldn[1], ldn[2])

    patches = []
    for point in points:
        patch = crop_landmark_patch(stldn, point)
        patches.append(patch.detach().cpu())

    feature_single = torch.stack(patches, dim=0).float()

    return feature_single


def select_uniform_frames(
    onset_frame: int, apex_frame: int, num_frames: int
) -> List[int]:
    """
    Uniformly select frame indices from onset to apex, then append apex + 1.

    Args:
        onset_frame:
            Onset frame number.

        apex_frame:
            Apex frame number.

        num_frames:
            Number of frames to select between onset and apex.

    Returns:
        frames:
            Selected frame numbers.

    Output length:
        num_frames + 1

    Reason:
        STLDN generation uses triplets. Adding apex + 1 allows the final
        triplet to include one frame after apex.
    """

    frames = []

    diff = (apex_frame - onset_frame) / num_frames

    for i in range(num_frames - 1):
        frames.append(onset_frame + ceil(diff * i))

    frames.append(apex_frame)
    frames.append(apex_frame + 1)

    return frames


def dataset_process(csv_path, image_root, num_classes, catego, num_frames=3):
    """
    Process a dataset and save STLDN-based model inputs.

    Args:
        csv_path:
            Path to the metadata CSV file.

        image_root:
            Root directory of extracted frames.

        num_classes:
            Number of emotion classes or "Folder".

        catego:
            Dataset name.

        num_frames:
            Number of selected frames.
            This value is overwritten by frame_num_selection(catego).

    Saves:
        data/processed_dataset/{catego}_processed.pkl

    Saved dictionary format:
        info[sample_key] = [
            feature_appearances,
            stldn_sequences.cpu()
        ]

    feature_appearances:
        Shape: [(T * 51), 7, 7]

    stldn_sequences:
        Shape: [T, H, W]
    """

    num_frames = frame_num_selection(catego)

    data, label_mapping = read_csv(csv_path, num_classes)
    emotion_categories = "Estimated Emotion " + str(num_classes)

    info = {}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kirsch_mask_detector = kirsch_mask_model().to(device)

    for idx in data.index:
        subject = data.loc[idx, "Subject"]
        onset_name = int(data.loc[idx, "OnsetFrame"])
        apex_name = int(data.loc[idx, "ApexFrame"])
        folder = data.loc[idx, "Filename"]

        # Read label for checking consistency.
        _ = label_mapping[data.loc[idx, emotion_categories]]

        print(
            f"idx={idx + 2}, subject={subject}, "
            f"folder={folder}, onset={onset_name}, apex={apex_name}"
        )

        frame_sequences = []
        stldn_sequences = []
        feature_coordinates = []
        feature_appearances = []

        frames = select_uniform_frames(onset_name, apex_name, num_frames)

        # ------------------------------------------------------------
        # Read previous and current frames
        # ------------------------------------------------------------
        previous_path = find_frame_path(catego, image_root, subject, folder, onset_name)
        current_path = find_frame_path(catego, image_root, subject, folder, frames[1])

        frame_previous = read_frame(previous_path, catego)
        frame_current = read_frame(current_path, catego)

        if catego == "SAMM":
            frame_previous = center_crop(frame_previous, (420, 420))
            frame_current = center_crop(frame_current, (420, 420))

        feature_coordinates.append(detect_landmarks(frame_previous))
        feature_coordinates.append(detect_landmarks(frame_current))

        frame_sequences.append(
            cv2.resize(frame_previous, dsize=(256, 256), interpolation=cv2.INTER_AREA)
            / 255.0
        )
        frame_sequences.append(
            cv2.resize(frame_current, dsize=(256, 256), interpolation=cv2.INTER_AREA)
            / 255.0
        )

        frame_current_face = frame_current

        frame_previous_tensor = torch.tensor(
            frame_previous, dtype=torch.float32, device=device
        )
        frame_current_tensor = torch.tensor(
            frame_current, dtype=torch.float32, device=device
        )

        ldn_previous = kirsch_mask_detector(frame_previous_tensor)
        ldn_current = kirsch_mask_detector(frame_current_tensor)

        # ------------------------------------------------------------
        # Generate STLDN maps from frame triplets
        # ------------------------------------------------------------
        for frame_name in frames[2:]:
            frame_path = find_frame_path(
                catego, image_root, subject, folder, frame_name
            )

            if not os.path.isfile(frame_path):
                frame_path = find_frame_path(
                    catego, image_root, subject, folder, frame_name - 1
                )

            frame_next = read_frame(frame_path, catego)

            if catego == "SAMM":
                frame_next = center_crop(frame_next, (420, 420))

            frame_sequences.append(
                cv2.resize(frame_next, dsize=(256, 256), interpolation=cv2.INTER_AREA)
                / 255.0
            )

            points = detect_landmarks(frame_current_face)
            frame_current_face = frame_next

            frame_next_tensor = torch.tensor(
                frame_next, dtype=torch.float32, device=device
            )
            ldn_next = kirsch_mask_detector(frame_next_tensor)

            stldn = generate_stldn_from_ldn(ldn_previous, ldn_current, ldn_next)
            stldn_sequences.append(stldn)

            patches = []
            for point in points:
                patch = crop_landmark_patch(stldn, point)
                patches.append(patch.detach().cpu())

            patches = torch.stack(patches, dim=0).float()

            ldn_previous = ldn_current
            ldn_current = ldn_next

            feature_coordinates.append(points)
            feature_appearances.append(patches)

        # ------------------------------------------------------------
        # Format appearance features
        # ------------------------------------------------------------
        feature_appearances = torch.stack(feature_appearances, dim=0)

        T, V, H, W = feature_appearances.size()

        feature_appearances = feature_appearances.view(T * V, H, W)

        # ------------------------------------------------------------
        # Format coordinate features
        # ------------------------------------------------------------
        feature_coordinates = np.array(feature_coordinates)
        feature_coordinates = torch.FloatTensor(feature_coordinates)

        T_coord, V_coord, W_coord = feature_coordinates.size()
        feature_coordinates = feature_coordinates.view(T_coord * V_coord, W_coord)

        # ------------------------------------------------------------
        # Format frame and STLDN sequences
        # ------------------------------------------------------------
        frame_sequences = torch.FloatTensor(np.array(frame_sequences))

        stldn_sequences = torch.stack(stldn_sequences, dim=0)

        # ------------------------------------------------------------
        # Optional apex frame processing
        # ------------------------------------------------------------
        apex_path = find_frame_path(catego, image_root, subject, folder, apex_name)
        apex_frame = read_frame(apex_path, catego, color_flag=1)

        if catego == "SAMM":
            apex_frame = center_crop(apex_frame, (420, 420))

        apex_frame = (
            cv2.resize(
                apex_frame,
                dsize=(224, 224),
                interpolation=cv2.INTER_AREA,
            )
            / 255.0
        )

        apex_frame = torch.FloatTensor(apex_frame).permute(2, 0, 1)

        # ------------------------------------------------------------
        # Save processed sample
        # ------------------------------------------------------------
        sample_key = f"{subject}_{folder}_{onset_name}"

        info[sample_key] = [
            feature_appearances,
            stldn_sequences.cpu(),
        ]

    os.makedirs("data/processed_dataset", exist_ok=True)

    output_path = f"data/processed_dataset/{catego}_processed.pkl"

    with open(output_path, "wb") as file:
        pickle.dump(info, file)

    print(f"--------- Done! Dataset size: {len(info)} ---------")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess frames into STLDN maps and landmark-centered STLDN patches."
    )

    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to the CSV file containing dataset metadata.",
    )

    parser.add_argument(
        "--image_root",
        type=str,
        required=True,
        help="Root directory containing extracted image frames.",
    )

    parser.add_argument(
        "--catego",
        type=str,
        required=True,
        help='Dataset name, for example "SAMM", "CAS(ME)^2", or "CAS(ME)^3".',
    )

    parser.add_argument(
        "--num_classes",
        type=str,
        default="Folder",
        help='Number of classes or "Folder" if labels are inferred from folders.',
    )

    args = parser.parse_args()

    dataset_process(
        csv_path=args.csv_path,
        image_root=args.image_root,
        num_classes=args.num_classes,
        catego=args.catego,
        num_frames=3,
    )
