import cv2
import dlib
import numpy as np
import util.face_alignment.align as face_align


def read_img(img_path):
    #read with dlib
    img = dlib.load_rgb_image(img_path)
    img_for_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #read with opencv
    # img = cv2.imread(img_path)[:,:,::-1]

    return img, img_for_show


def landmark_detection(img_path, face_detector, landmark_detector):
    img, img_for_show = read_img(img_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 1) # upscale: 한 번을 하면 큰 이미지를 인식함.

    landmark_list = []
    RIGHT_EYE = list(range(36, 42))
    LEFT_EYE = list(range(42, 48))
    EYES = list(range(36, 48))

    for k, d in enumerate(faces):  # 얼굴 마다
        landmarks = landmark_detector(img, d)

        # 오른쪽 눈
        x_right_sum = 0
        y_right_sum = 0
        for n in RIGHT_EYE:
            x_right_sum += landmarks.part(n).x
            y_right_sum += landmarks.part(n).y

        x_right = int(x_right_sum / 6)
        y_right = int(y_right_sum / 6)
        right_eye_center = [x_right, y_right]
        landmark_list.append(right_eye_center)
        cv2.circle(img_for_show, (x_right, y_right), 2, (255, 255, 0), -1)

        # 왼쪽 눈
        x_left_sum = 0
        y_left_sum = 0
        for n in LEFT_EYE:
            x_left_sum += landmarks.part(n).x
            y_left_sum += landmarks.part(n).y

        x_left = int(x_left_sum / 6)
        y_left = int(y_left_sum / 6)
        left_eye_center = [x_left, y_left]
        landmark_list.append(left_eye_center)
        cv2.circle(img_for_show, (x_left, y_left), 2, (255, 255, 0), -1)

        # 코: 33, 입, 48, 54
        for n in [33, 48, 54]:
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmark_list.append([x, y])
            cv2.circle(img_for_show, (x, y), 2, (255, 255, 0), -1)

    landmark_array = np.array(landmark_list)
    return landmark_array, img_for_show


def face_alignment(img_path, face_detector, landmark_detector):
    img_landmark_array, img_for_show = landmark_detection(img_path, face_detector, landmark_detector)
    try:
        wraped = face_align.norm_crop(img_for_show, img_landmark_array)
    except AssertionError:
        return None
    return wraped

