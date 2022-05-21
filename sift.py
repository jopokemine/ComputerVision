import numpy as np
import cv2 as cv

img = cv.imread("0.png")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
kp, des = sift.detectAndCompute(gray, None)

img = cv.drawKeypoints(gray, kp, img)
cv.imwrite("sift_keypoints.jpg", img)
