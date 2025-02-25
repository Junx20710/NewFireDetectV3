import cv2
import time

import threading

from flask import Flask, request, Response, render_template
import cv2
import torch
import time
import datetime
import os
from fire_detection import fire_detect,fire_detect_json
import json
import queue

# 全局变量
detection_queue = queue.Queue()
is_rendering = False


app = Flask(__name__)

def detectFire(stream_url):
    if not stream_url:
        return "No stream URL provided", 400

    return fire_detect_json(video_url=stream_url)


video_path = "rtsp://127.0.0.1:8554/video"
A=detectFire(video_path)

'''
if __name__ == '__main__':
    # 无人机rtmp视频流url
    video_path = "rtsp://127.0.0.1:8554/123"

    # opencv获取视频流
    print("-----正在进行前期准备-----")
    capture = cv2.VideoCapture(video_path,cv2.CAP_FFMPEG)
    print("-----正在进行前期准备-----")
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 设置缓冲区大小为1帧（减少延迟）
    capture.set(cv2.CAP_PROP_POS_FRAMES, 5000)  # 设置某个开始帧，避免长时间读取

    print("-----正在获取视频流-----")
    while True:
        # fps = 0.0

        t1 = time.time()
        # 读取某一帧
        ref, frame = capture.read()
        # fps = (fps + (1. / (time.time() - t1))) / 2
        # print("fps= %.2f" % (fps))

        if frame is None or frame.size == 0:
            print("Empty frame received.")
            continue

        # frame = cv2.putText(frame)

        cv2.imshow("video", frame)

        c = cv2.waitKey(1) & 0xff
        if c == 27:
            capture.release()
            break
    print("Video Detection Done!")
    capture.release()
    cv2.destroyAllWindows()
'''