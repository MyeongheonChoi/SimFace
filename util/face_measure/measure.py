import numpy as np


def gonion_eye_angle(left_gonion, left_eye):
    # 3차원 정보가 있으면 atan으로 구할 수 있을 것 같긴 한데 우선 단순히 거리로 계산
    return distance(left_gonion, left_eye[[0]])


def eye_nasion_distnace(left_eye, nasion):
    return distance(nasion, left_eye.mean(axis=0))


def inter_eye_width(right_eye, left_eye):
    eye2eye_distance = np.min(right_eye[:, 0]) - np.max(left_eye[:, 0])
    return eye2eye_distance


def eye_shape(left_eye):
    left_i_area = slope(left_eye.mean(axis=0), left_eye[0])
    return left_i_area


def nose_height(noseLine):
    noseheight = np.max(noseLine[:, 1])- np.min(noseLine[:, 1])
    return noseheight


def inter_tragi(facial_points):
    inter_tragi_distance = facial_points[16][0] - facial_points[0][0]
    return inter_tragi_distance


def mid_face_height(right_ibrow, nose_line):
    right_ibrow_y = right_ibrow[2, 1]
    middle_face_height = nose_line[3, 1] - right_ibrow_y  # << Final result
    return  middle_face_height


def bridge_of_nose(noseArc):
    nosebridge = noseArc[3][0] - noseArc[0][0]
    return nosebridge


# Geometry
def distance(a, b):
    return np.sqrt(np.sum((a-b)**2))


def shape_area(points, circularArray=False):
    # Function to calculate area of any shape given its points coordinates
    # Circular array means that first point is added to the end of the array
    result = 0
    for i in range(len(points)-1):
        x1, y1 = points[i]
        x2, y2 = points[i+1]
        result += (x1*y2) - (y1*x2)
    if not circularArray:
        x1, y1 = points[len(points)-1]
        x2, y2 = points[0]
        result += (x1*y2) - (y1*x2)
    result /= 2
    return abs(result)


def slope(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    deltaX = x2-x1
    deltaY = y2-y1
    if deltaX == 0:
        return 0
    slope = deltaY / deltaX
    return round(slope, 3)

