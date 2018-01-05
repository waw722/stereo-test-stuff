import cv2
import numpy
import yaml

CORNER_SUBPIX_WINDOW_SIZE = (11, 11)
CORNER_SUBPIX_TERMINATION_CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001
)


def _create_obj_grid(size, scale):
    res = numpy.zeros((size[0] * size[1], 3), numpy.float32)
    for i in range(size[1]):
        for j in range(size[0]):
            res[j + i * size[0]] = (j * scale, i * scale, 0)
    return res


def dump_conf(conf_file, data):
    with open(conf_file, 'w') as fp:
        yaml.dump(data, fp)


def load_conf(conf_file):
    with open(conf_file, 'r') as fp:
        data = yaml.load(fp)
    return data


def crop_to_roi(img, roi):
    x, y, w, h = roi
    return img[y:y + h, x:x + w]


def calculate_dist_mtx(obj_points, img_points, img_shape):
    data = {}
    h, w, c = img_shape
    ret, data['mtx'], data['dist'], rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (h, w), None, None)
    data['opt_mtx'], data['roi'] = cv2.getOptimalNewCameraMatrix(
        data['mtx'], data['dist'], (w, h), 0, (w, h))
    return data


def live_calibrate(conf_file, vid_cap, chessboard_size, scale):
    img_points = []
    while len(img_points) < 12:
        ret, img = vid_cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if not ret:
            img = cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(1)
            continue
        corners = cv2.cornerSubPix(
            gray, corners, CORNER_SUBPIX_WINDOW_SIZE, (-1, -1),
            CORNER_SUBPIX_TERMINATION_CRITERIA)
        img_points.append(corners)
        img = cv2.drawChessboardCorners(img, chessboard_size, corners, True)
        cv2.imshow('img', img)
        cv2.waitKey(500)
    cv2.destroyAllWindows()
    obj_points = [_create_obj_grid(chessboard_size, scale)] * len(img_points)
    data = calculate_dist_mtx(obj_points, img_points, img.shape)
    dump_conf(conf_file, data)
    return data


def file_calibrate(conf_file, img_paths, chessboard_size, scale):
    img_points = []
    for path in img_paths:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            corners = cv2.cornerSubPix(
                gray, corners, CORNER_SUBPIX_WINDOW_SIZE, (-1, -1),
                CORNER_SUBPIX_TERMINATION_CRITERIA)
            img_points.append(corners)
    obj_points = [_create_obj_grid(chessboard_size, scale)] * len(img_points)
    data = calculate_dist_mtx(obj_points, img_points, img.shape)
    dump_conf(conf_file, data)
    return data


def undistort(data, img):
    res = cv2.undistort(img, data['mtx'], data['dist'], None, data['opt_mtx'])
    x, y, w, h = data['roi']
    res = res[y:y + h, x:x + w]
    return res
