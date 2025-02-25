from flask import Flask, request, Response, jsonify, render_template
import cv2
import threading
import json
from fire_detection import fire_detect_json
from collections import deque
import torch
import sys
import os
# 将 `fire` 目录添加到 Python 模块搜索路径中
sys.path.append(os.path.abspath('fire'))
from fire.models.experimental import attempt_load
from fire.utils.general import (
    check_img_size, non_max_suppression, scale_coords, plot_one_box)
from fire.utils.torch_utils import select_device, time_synchronized
from fire.utils.datasets import LoadStreams, LoadImages, letterbox


app = Flask(__name__)

# ffmpeg -re -stream_loop -1 -i fire.mp4 -c copy -f rtsp rtsp://127.0.0.1:8554/video

# Global variable to store the detections
detections = deque(maxlen=20)  # Reduce the size of the queue

@app.route('/')
def index():
    return render_template('index.html')

weights='fire/best.pt'
img_size=640
conf_thres=0.4
iou_thres=0.5
device='cpu'
view_img=True
output='output'
import numpy as np

def detect(frame, imgs, timestampNow,model):
    s = np.stack([letterbox(x, new_shape= img_size)[0].shape for x in imgs], 0)  # inference shapes
    rect = np.unique(s, axis=0).shape[0] == 1

    im0s = imgs.copy()

    # Letterbox
    img = [letterbox(x, new_shape=img_size, auto=rect)[0] for x in im0s]

    # Stack
    img = np.stack(img, 0)

    # Convert
    img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
    img = np.ascontiguousarray(img)
    
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    t2 = time_synchronized()
    print("-----写入图像-----")
    output_file_path = 'output/res.txt'  # 结果文件路径

    # 确保输出文件可写入
    with open(output_file_path, 'a') as output_file:  # 以追加模式打开文件
        frame_results = []  # 存储当前帧的检测结果

        for i, det in enumerate(pred):
            im0 = im0s[i].copy()
            frame_info = {
                "cls": [],
                "timestamp": timestampNow,
                "detections": []
            }

            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in det:
                    bbox = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                    frame_info["detections"].append({"bbox": bbox})
                    frame_info["cls"].append(int(cls))  # 添加类到 frame_info

                # 将完整的 frame_info 写入文件
                # output_file.write(f'{json.dumps(frame_info)}\n')  # 将 frame_info 转为JSON格式写入

            # 只在有检测结果时添加到结果列表
            if frame_info["detections"]:
                frame_results.append(frame_info)
        return frame_results


def generate_frames(stream_url):
    global detections

    cap = cv2.VideoCapture(stream_url)
    
    ret, frame = cap.read()
    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)

    # Initialize the fire detection generator
    print("-----初始化-----")
    device = select_device('cpu')
    model = attempt_load(weights, map_location=device)
    half = device.type != 'cpu'

    if half:
        model.half()

    print("-----准备模型-----")
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[255, 255, 0], [0, 255, 0]]  # 火焰蓝色，烟雾绿色
        
    print("-----准备模型V2-----")
    # 预热模型
    img = torch.zeros((1, 3, img_size, img_size), device=device)
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    
    flag = True
    frame_results = []
    imgs = []
    cnt = 0
    
    while True:
        ret, frame = cap.read()
        imgs.append(frame)
        if not ret:
            flag = False
            break
        cnt += 1
        cnt %= 5
        timestampNow = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        if cnt == 0:
            frame_results = detect(frame, imgs, timestampNow,model)
            imgs = []
        
        # Draw the detections on the frame
        for detection in frame_results:
           for bbox in detection['detections']:
                start_point = (bbox['bbox'][0], bbox['bbox'][1])
                end_point = (bbox['bbox'][2], bbox['bbox'][3])
                color = (0, 0, 255)  # BGR color for red
                thickness = 2
                frame = cv2.rectangle(frame, start_point, end_point, color, thickness)
                
        # Store the detections
        detections.append(frame_results)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)

        # Convert to bytes and yield as a frame
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    stream_url = request.args.get('stream_url', '')
    return Response(generate_frames(stream_url), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detections')
def get_detections():
    global detections
    return jsonify(list(detections))

if __name__ == '__main__':
    app.run(threaded=True, host='0.0.0.0', port=5000)
