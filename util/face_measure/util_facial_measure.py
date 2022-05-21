# Reference
# https://github.com/zementalist/Facial-Features-Measurement-and-Analysis/blob/main/scripts/feature_analysis.py
#

import numpy as np
import pandas as pd
import cv2
from imutils import face_utils
import util.face_measure.measure as measure


def load_facial_components(facial_points):
    # Function to collect landmarks points, grouped, as polygons\shapes
    # TODO: 사용할 facial points 정리하기
    nasion = facial_points[[28]]
    zygoma = facial_points[[1]]
    gonion = facial_points[[5, 6]].mean(axis=0)

    left_eye = np.concatenate((facial_points[36:42], np.array([facial_points[36]])))
    rightEye = np.concatenate((facial_points[42:47], np.array([facial_points[42]])))
    leftIBrow = facial_points[17:22]
    rightIBrow = facial_points[22:27]
    noseLine = facial_points[27:31]
    noseArc = facial_points[31:36]
    upperLip = facial_points[[50, 51, 52, 63, 62, 61, 50]]
    lowerLip = facial_points[[67, 66, 65, 56, 57, 58, 67]]
    faceComponents = {
        'nasion' : nasion,
        'zygoma': zygoma,
        'gonion': gonion,
        "left_eye": left_eye,
        "right_eye": rightEye,
        "left_i_brow": leftIBrow,
        "right_i_brow": rightIBrow,
        "nose_line": noseLine,
        "nose_arc": noseArc,
        "upper_lip": upperLip,
        "lower_lip": lowerLip
    }
    return faceComponents


def measure_features(facial_points):
    faceComponents = load_facial_components(facial_points)
    left_eye, right_eye = faceComponents["left_eye"], faceComponents["right_eye"]
    left_ibrow, right_ibrow = faceComponents["left_i_brow"], faceComponents["right_i_brow"]
    nose_line, nose_arc = faceComponents["nose_line"], faceComponents["nose_arc"]

    # TODO: measure할 feature 정리하기
    bridge_of_nose = measure.bridge_of_nose(nose_arc)

    eye_shape = measure.eye_shape(left_eye)
    eye_nasion_distnace = measure.eye_nasion_distnace(left_eye, faceComponents['nasion'])
    gonion_eye_angle = measure.gonion_eye_angle(faceComponents['gonion'], left_eye)
    inter_eye_width = measure.inter_eye_width(left_eye, right_eye)

    mid_face_height = measure.mid_face_height(right_ibrow, nose_line)
    inter_tragi = measure.inter_tragi(facial_points)

    measures = {
        "bridge_of_nose" : bridge_of_nose,
        "eye_shape" : eye_shape,
        "eye_nasion_distnace" : eye_nasion_distnace,
        "gonion_eye_angle" :gonion_eye_angle,
        "inter_eye_width" : inter_eye_width,
        "mid_face_height" : mid_face_height,
        "inter_tragi" : inter_tragi,
        }

    return pd.Series(measures, name="Face features measures")


def face_landmarks(img, face_detector, landmark_detector):
    grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rectangles = face_detector(grayscale_image, 1)

    if len(rectangles) == 0:
        return None
    landmarks = landmark_detector(grayscale_image, rectangles[0])
    landmarks = face_utils.shape_to_np(landmarks)

    return landmarks
