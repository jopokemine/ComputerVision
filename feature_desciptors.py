import numpy as np
import cv2 as cv
import os


def generate_sift(img_path, out_path=None):
    img = cv.imread(img_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)

    img = cv.drawKeypoints(gray, kp, img)
    if out_path != None:
        cv.imwrite(os.path.join(
            out_path, f'{os.path.split(img_path)[-1][:-4]}_sift.png'), img)
    return des


def generate_orb(img_path, out_path=None):
    img = cv.imread(img_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    orb = cv.ORB_create()
    kp = orb.detect(gray, None)
    kp, des = orb.compute(gray, kp)

    img = cv.drawKeypoints(gray, kp, img)
    if out_path != None:
        cv.imwrite(os.path.join(
            out_path, f'{os.path.split(img_path)[-1][:-4]}_orb.png'), img)
    return des
