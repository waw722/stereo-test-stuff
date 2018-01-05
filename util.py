import cv2
import contextlib
import numpy as np
import textwrap


@contextlib.contextmanager
def time_printer(name):
    start = cv2.getTickCount()
    yield
    end = cv2.getTickCount()
    time = (end - start) / cv2.getTickFrequency()
    print('{}:\t{}s'.format(name, time))


def cv_wait_until_key(key):
    keycode = ord(key[0]) if isinstance(key, str) else key
    while True:
        if cv2.waitKey() & 0xFF == keycode:
            break


def write_ply(target, points, colors):
    ply_header = textwrap.dedent('''
        ply
        format ascii 1.0
        element vertex {point_num}
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        ''')[1:]
    points = points.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    points = np.hstack([points, colors])
    with open(target, 'wb') as fp:
        fp.write(ply_header.format(point_num=len(points)).encode())
        np.savetxt(fp, points, fmt="%f %f %f %d %d %d ")
