#!/usr/bin/python3

import cv2
import calib
import sys


def show_camera(vc, calibration_data):
    while True:
        ret, img = vc.read()
        cv2.imshow("original", img)
        res = calib.undistort(calibration_data, img)
        cv2.imshow("undistored", res)
        if (cv2.waitKey(20) & 0xFF) == 0x1B:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("expect calibration data file and camera index")
    calibration_data = calib.load_conf(sys.argv[1])
    camera_index = int(sys.argv[2])
    vc = cv2.VideoCapture(camera_index)
    show_camera(vc, calibration_data)
