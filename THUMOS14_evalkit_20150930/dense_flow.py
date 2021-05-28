import numpy as np
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

def cal_dense_flow(frame1, frame2):
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    next_frame = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def gen_flow_video(video_path):
    new_vid_path = video_path.replace('INFRARED', 'denseflow')
    new_dir = os.path.split(new_vid_path)[0]
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(3))
    h = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(new_vid_path, fourcc, 25, (w, h))
    ret, prvs = cap.read()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        flow = cal_dense_flow(prvs, frame)
        out.write(flow)
        prvs = frame
    out.release()
    cap.release()

if __name__ == "__main__":
    dir_name = '/raid/hukai/INFRARED'
    for parentdir, dirnames, filenames in os.walk(dir_name):
        for filename in tqdm(filenames):
            if '.avi' in filename:
                video_path = os.path.join(parentdir, filename)
                gen_flow_video(video_path)