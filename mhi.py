import numpy as np
import cv2 as cv
import os

MHI_DURATION = 50
DEFAULT_THRESHOLD = 32


def generate_mhi(video_path, out_path=None, every_n_frames=0) -> np.uint8:
    cam = cv.VideoCapture(video_path)
    ret, frame = cam.read()
    h, w = frame.shape[:2]
    prev_frame = frame.copy()
    motion_history = np.zeros((h, w), np.float32)
    timestamp = 0
    idx = 0
    frame_count = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        frame_diff = cv.absdiff(frame, prev_frame)
        gray_diff = cv.cvtColor(frame_diff, cv.COLOR_BGR2GRAY)
        ret, fgmask = cv.threshold(
            gray_diff, DEFAULT_THRESHOLD, 1, cv.THRESH_BINARY)
        timestamp += 1

        # update motion history
        cv.motempl.updateMotionHistory(
            fgmask, motion_history, timestamp, MHI_DURATION)

        # normalize motion history
        mh = np.uint8(np.clip(
            (motion_history - (timestamp - MHI_DURATION)) / MHI_DURATION, 0, 1) * 255)

        prev_frame = frame.copy()

        frame_count += 1
        if out_path != None and frame_count == every_n_frames:
            frame_count = 0
            idx += 1
            cv.imwrite(os.path.join(
                out_path, f'{os.path.split(video_path)[-1][:-4]}_{idx}_mhi.png'), mh)

    if out_path != None:
        cv.imwrite(os.path.join(
            out_path, f'{os.path.split(video_path)[-1][:-4]}_{idx + 1}_mhi.png'), mh)
    return mh
