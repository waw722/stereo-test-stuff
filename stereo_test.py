#!/usr/bin/python3

import cv2
import numpy as np
import sys
import util

STEREO_SGBM_MIN_DISP = 16
STEREO_SGBM_NUM_DISP = (16 * 9) - STEREO_SGBM_MIN_DISP
STEREO_SGBM_WINDOW_SIZE = 3
STEREO_SGBM = cv2.StereoSGBM_create(
    minDisparity=STEREO_SGBM_MIN_DISP,
    numDisparities=STEREO_SGBM_NUM_DISP,
    blockSize=5,
    P1=8 * 3 * STEREO_SGBM_WINDOW_SIZE ** 2,
    P2=32 * 3 * STEREO_SGBM_WINDOW_SIZE ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=8,
    speckleWindowSize=200,
    speckleRange=2,
    # mode=cv2.STEREO_SGBM_MODE_HH,
)
STEREO_BM = cv2.StereoBM_create(numDisparities=16, blockSize=11)


def view_disp(disp, name):
    disp_view = ((disp - disp.min()) / disp.max())
    cv2.imshow(name, disp_view)
    util.cv_wait_until_key('q')
    cv2.destroyAllWindows()


def rectify_to_ply(target, disp, color_img):
    h, w = color_img.shape[:2]
    f = 0.8 * w  # guess for focal length
    Q = np.float32([[1, 0, 0, -0.5 * w],
                    [0, -1, 0, 0.5 * h],
                    [0, 0, 0, -f],
                    [0, 0, 1, 0]])
    mask = disp > disp.min()
    points = cv2.reprojectImageTo3D(disp, Q)[mask]
    colors = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)[mask]
    util.write_ply(target, points, colors)


if __name__ == "__main__":
    ply_file = '/tmp/disparity_{}.ply'
    # algos = (STEREO_BM, STEREO_SGBM)
    algos = (STEREO_SGBM,)
    # input images must be undistorted
    if len(sys.argv) != 3:
        print("expect left and right image paths")
        sys.exit(1)
    left = cv2.imread(sys.argv[1], 0)
    right = cv2.imread(sys.argv[2], 0)
    left_color = cv2.imread(sys.argv[1])
    for algo in algos:
        with util.time_printer(type(algo).__name__):
            disp = algo.compute(left, right)
        view_disp(disp, type(algo).__name__)
        rectify_to_ply(ply_file.format(type(algo).__name__), disp, left_color)
