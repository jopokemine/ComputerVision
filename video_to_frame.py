import os
import numpy as np
import cv2 as cv
from glob import glob

def save_frame(video_path, save_dir):
    name = video_path.split("/")[-1].split(".")[0]
    save_path = os.path.join(save_dir, name)
    create_dir(save_path)

    cap = cv.VideoCapture(video_path)
    idx = 0

    while True:
        ret, frame = cap.read()

        if ret == False:
            cap.release()
            break
        
        cv.imwrite(f"{save_path}/{idx}.png", frame)

        idx += 1


def create_dir(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"Error: Creating directory {path}")


if __name__ == "__main__":
    video_paths = glob("database/videos/*")
    save_dir = "database/frames"

    for path in video_paths:
        save_frame(path, save_dir)
