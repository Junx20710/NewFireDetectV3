import cv2
import time
import torch
import sys
import os
# 将 `fire` 目录添加到 Python 模块搜索路径中
sys.path.append(os.path.abspath('fire'))
from fire.models.experimental import attempt_load
from fire.utils.general import (
    check_img_size, non_max_suppression, scale_coords, plot_one_box)
from fire.utils.torch_utils import select_device, time_synchronized
from fire.utils.datasets import LoadStreams, LoadImages
import json

import base64
import json
import cv2
import torch
from flask import Response

def fire_detect_json(video_url="rtmp://192.168.x.x:1935/live", weights='fire/best.pt', img_size=640,
                conf_thres=0.4, iou_thres=0.5, device='cpu', view_img=True, output='output'):
    # 初始化设备和模型
    print("-----初始化-----")
    device = select_device(device)
    model = attempt_load(weights, map_location=device)
    half = device.type != 'cpu'

    if half:
        model.half()

    print("-----准备模型-----")
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[255, 255, 0], [0, 255, 0]]  # 火焰蓝色，烟雾绿色

    # 初始化视频流捕获
    capture = LoadStreams(video_url, img_size=img_size)
    if not capture:
        print("视频流捕获失败，检查 stream_url")
        
    print("-----准备模型V2-----")
    # 预热模型
    img = torch.zeros((1, 3, img_size, img_size), device=device)
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    print("-----开始检测-----")
    while True:
        try:
            for path, img, im0s, vid_cap in capture:
                # 检查是否捕获到图像数据
                if img is None or im0s is None:
                    print("捕获的帧为空，跳过当前帧...")
                    continue  # 跳过空帧

                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()
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

                        if vid_cap is not None:
                            frame_info = {
                                "cls": [],
                                "timestamp": max(vid_cap),
                                "detections": []
                            }
                        else:
                            continue

                        if det is not None and len(det):
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                            for *xyxy, conf, cls in det:
                                bbox = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                                frame_info["detections"].append({"bbox": bbox})
                                frame_info["cls"].append(int(cls))  # 添加类到 frame_info

                            # 将完整的 frame_info 写入文件
                            output_file.write(f'{json.dumps(frame_info)}\n')  # 将 frame_info 转为JSON格式写入

                        # 只在有检测结果时添加到结果列表
                        if frame_info["detections"]:
                            frame_results.append(frame_info)

                    frame_results = sorted(frame_results, key=lambda x: x['timestamp'])
                    # 返回每帧的检测结果为JSON
                    yield json.dumps(frame_results) if frame_results else json.dumps([])  # 返回有效结果或空列表

        except Exception as e:
            print(f"检测过程中出错：{e}")
            # 如果流断开或者出错，可以尝试重新连接
            capture = LoadStreams(video_url, img_size=img_size)
            print("重新连接视频流...")
            if not capture:
                print("视频流捕获失败，检查 stream_url")
            continue  # 重试下一次循环


def fire_detect(video_url="rtmp://192.168.x.x:1935/live", weights='fire/best.pt', img_size=640,
                conf_thres=0.4, iou_thres=0.5, device='cpu', view_img=True, output='output'):
    # 初始化设备和模型
    print("-----初始化-----")
    device = select_device(device)
    model = attempt_load(weights, map_location=device)
    half = device.type != 'cpu'

    if half:
        model.half()

    print("-----准备模型-----")
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[255, 255, 0], [0, 255, 0]] # 火焰蓝色，烟雾绿色

    # 初始化视频流捕获
    capture = LoadStreams(video_url, img_size=img_size)
    if not capture:
        print("视频流捕获失败，检查 stream_url")

    # 预热模型
    img = torch.zeros((1, 3, img_size, img_size), device=device)
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    print("-----开始检测-----")
    while True:
        try:
            for path, img, im0s, vid_cap in capture:
                # 检查是否捕获到图像数据
                if img is None or im0s is None:
                    print("捕获的帧为空，跳过当前帧...")
                    continue  # 跳过空帧

                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()
                img /= 255.0  # 0 - 255 to 0.0 - 1.0

                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Inference
                t1 = time_synchronized()
                pred = model(img, augment=False)[0]
                pred = non_max_suppression(pred, conf_thres, iou_thres)
                t2 = time_synchronized()

                print("-----写入图像-----")
                for i, det in enumerate(pred):
                    im0 = im0s[i].copy()

                    if det is not None and len(det):
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        for *xyxy, conf, cls in det:
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                    # 编码成JPEG格式并通过yield返回
                    ret, buffer = cv2.imencode('.jpg', im0)
                    if ret:
                        frame = buffer.tobytes()

                        # 通过yield每帧发送给前端
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    else:
                        print("编码帧失败，跳过当前帧。")

        except Exception as e:
            print(f"检测过程中出错：{e}")
            # 如果流断开或者出错，可以尝试重新连接
            capture = LoadStreams(video_url, img_size=img_size)
            print("重新连接视频流...")
            continue  # 重试下一次循环



