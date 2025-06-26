import argparse
import numpy as np
import dlib
# import glob
# import pandas as pd
# import cv2


# Place for the pretrained old model for dlib
CNN_FACE_MODEL_PATH = "weight/mmod_human_face_detector.dat"
# PREDICTOR_PATH = "weight/shape_predictor_68_face_landmarks.dat"
PREDICTOR_PATH = "data_utilz/weight/shape_predictor_68_face_landmarks.dat"

#MMOD CNN: dlib.cnn_face_detection_model_v1(modelPath)

# Initialize the predictor and face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def convert(landmarks):
    result = []
    for points in landmarks.parts():
        result.append((points.x, points.y))

    return result


def detect_landmarks(img):
    # Predict the 68 landmarks and convert to list
    detect_faces = detector(img, 1)

    for detect_face in detect_faces:
        landmarks = predictor(img, detect_face)

    landmarks = convert(landmarks)

    # Not taking the landmarks at the edge of the face
    lmarks = np.array(landmarks)[17:, :]

    # if self.train: # augmentation
    #     sigma = random.uniform(0, 5)
    #     lmarks = lmarks + np.random.randn(*lmarks.shape) * sigma
    #     sigma = random.uniform(0, 10)
    #     img = img + np.random.randn(*img.shape) * (sigma / 255.)


    return lmarks


def save_landmarks_csv(output_path):
    final_landmarks = []

    img_generator = glob.glob("*.jpg")
    for img_path in img_generator:
        landmarks = detect_landmarks(img_path)
        final_landmarks.append(landmarks)

    landmarks_csv = pd.DataFrame(final_landmarks)
    landmarks_csv.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_path", type=str, default= "D:/Datasets/Image/CAS(ME)^2/casme/cropped/cropped/15/anger1_1/img572.jpg")
    parser.add_argument("--csv", type=str) #required=True
    parser.add_argument("--path", type=str) #required=True
    parser.add_argument("--output", type=str) #required=True
    args = parser.parse_args()

    frame = cv2.imread(args.frame_path)
    print(frame.shape)
    points = detect_landmarks(frame)
    print(points.shape)

    # save_landmarks_csv(args.output)
