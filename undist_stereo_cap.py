#!/usr/bin/python3

import calib
import cv2
import os
import sys
import time


def capture(data0, data1, vc0, vc1, target_dir, index):
    vc0.grab()
    vc1.grab()
    _, img0 = vc0.retrieve()
    _, img1 = vc1.retrieve()
    img0 = calib.undistort(data0, img0)
    img1 = calib.undistort(data1, img1)
    cv2.imwrite(os.path.join(target_dir, '{}-img0.jpg'.format(index)), img0)
    cv2.imwrite(os.path.join(target_dir, '{}-img1.jpg'.format(index)), img1)


if __name__ == "__main__":
    if len(sys.argv) != 6:
        raise ValueError("expect calibration data files and camera indices")
    target = sys.argv[1]
    data0 = calib.load_conf(sys.argv[2])
    data1 = calib.load_conf(sys.argv[3])
    vc0 = cv2.VideoCapture(int(sys.argv[4]))
    vc1 = cv2.VideoCapture(int(sys.argv[5]))
    for _ in range(5):
        vc0.read()
        vc1.read()
        time.sleep(0.1)
    for i in range(10):
        capture(data0, data1, vc0, vc1, target, i)
        time.sleep(0.2)
