#!/usr/bin/env python3
"""
The code is edited from docs (https://docs.luxonis.com/projects/api/en/latest/samples/Yolo/tiny_yolo/)
We add parsing from JSON files that contain configuration
"""

from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time
import argparse
import json
import blobconverter
import math
from PIL import Image, ImageDraw, ImageFont
import os

full_area = 32*32
width_window = 25
dist_to_camera = 53.5
wheigth_1_1_sphere = 0.33
r = 0.5
v = 4/3*math.pi *r**3
ro = wheigth_1_1_sphere/v

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Provide model name or model path for inference",
                    default='yolov4_tiny_coco_416x416', type=str)
parser.add_argument("-c", "--config", help="Provide config path for inference",
                    default='json/yolov4-tiny.json', type=str)
args = parser.parse_args()

# parse config
configPath = Path(args.config)
if not configPath.exists():
    raise ValueError("Path {} does not exist!".format(configPath))

with configPath.open() as f:
    config = json.load(f)
nnConfig = config.get("nn_config", {})

# parse input shape
if "input_size" in nnConfig:
    W, H = tuple(map(int, nnConfig.get("input_size").split('x')))

# extract metadata
metadata = nnConfig.get("NN_specific_metadata", {})
classes = metadata.get("classes", {})
coordinates = metadata.get("coordinates", {})
anchors = metadata.get("anchors", {})
anchorMasks = metadata.get("anchor_masks", {})
iouThreshold = metadata.get("iou_threshold", {})
confidenceThreshold = metadata.get("confidence_threshold", {})




print(metadata)

# parse labels
nnMappings = config.get("mappings", {})
labels = nnMappings.get("labels", {})

# get model path
nnPath = args.model
if not Path(nnPath).exists():
    print("No blob found at {}. Looking into DepthAI model zoo.".format(nnPath))
    nnPath = str(blobconverter.from_zoo(args.model, shaves = 6, zoo_type = "depthai", use_cache=True))
# sync outputs
syncNN = True

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
xoutRgb = pipeline.create(dai.node.XLinkOut)
nnOut = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
nnOut.setStreamName("nn")

# Properties
camRgb.setPreviewSize(W, H)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFps(40)

# Network specific settings
detectionNetwork.setConfidenceThreshold(confidenceThreshold)
detectionNetwork.setNumClasses(classes)
detectionNetwork.setCoordinateSize(coordinates)
detectionNetwork.setAnchors(anchors)
detectionNetwork.setAnchorMasks(anchorMasks)
detectionNetwork.setIouThreshold(iouThreshold)
detectionNetwork.setBlobPath(nnPath)
detectionNetwork.setNumInferenceThreads(2)
detectionNetwork.input.setBlocking(False)

# Linking
camRgb.preview.link(detectionNetwork.input)
detectionNetwork.passthrough.link(xoutRgb.input)
detectionNetwork.out.link(nnOut.input)

# Function to put text on frame using Pillow
def putText(img, text, position, font_size, color):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    font_path = "/home/gasan/PycharmProjects/depthAI_main/test/gen2-yolo/device-decoding/fonts/font_kirill.ttf"  # Укажите путь к вашему шрифту
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    img = np.array(img_pil)
    return img

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output3.mp4', fourcc,20.0, (W, H))

result_vid = cv2.VideoWriter('result_vid3.mp4', fourcc,20.0, (W, H))
# Create images directory if it doesn't exist
os.makedirs("images", exist_ok=True)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    frame = None
    detections = []
    startTime = time.monotonic()
    counter = 0
    color2 = (255, 255, 255)

    save_time = startTime
    save_interval = 0.5  # Save every 0.5 seconds

    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def displayFrame(name, frame, detections, save_time):

        color = (255, 0, 0)
        sum_weigth = 0
        farme_shape = frame.shape
        area_pixel = farme_shape[0]*farme_shape[1]
        des = full_area/area_pixel

        black_frame = np.zeros(farme_shape, dtype=int)
        dict_text_array = []
        full_area_frame = 0
        for detection in detections:
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            x1, y1, x2, y2 = bbox
            roi = frame[y1:y2, x1:x2]

            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            l_green = np.array([40, 0, 0])
            u_green = np.array([255, 255, 255])

            mask = cv2.inRange(hsv, l_green, u_green)

            kernels = np.ones((5, 5), np.uint8)
            mask = cv2.erode(mask, kernels, iterations=1)

            bg = np.ones(roi.shape, dtype=np.uint8) * 255
            black = np.zeros(roi.shape, dtype=np.uint8)

            mask_inv = cv2.bitwise_not(mask)

            fg = cv2.bitwise_and(roi, roi, mask=mask)
            bg = cv2.bitwise_and(bg, bg, mask=mask_inv)

            result = cv2.add(fg, bg)

            top, bottom = 40, 40
            left, right = 40, 40
            black_border = cv2.copyMakeBorder(black, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

            black_border_resized = cv2.resize(black_border, (result.shape[1], result.shape[0]))
            result_with_border = cv2.add(result, black_border_resized)

            result_with_border_gray = cv2.cvtColor(result_with_border, cv2.COLOR_BGR2GRAY)

            _, thresh = cv2.threshold(result_with_border_gray, 248, 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:
                    contour_offset = contour + (x1, y1)
                    cv2.drawContours(frame, [contour_offset], -1, (255, 255, 255), 2)
                    dict_text_array.append({
                        "text": f'S={round(area * des, 2)}см2.',
                        "coord": (x1, y1 - 20),
                        "font_size": 25,
                        "color": (255, 255, 255)
                    })
                    full_area_frame += round(area * des, 2)
        for text in dict_text_array:
            frame = putText(frame, text['text'], text['coord'], text['font_size'], text['color'])
        frame = putText(frame, f"Количество обьектов: {len(detections)}", (10, 30), 25  , (255,255,255))
        frame = putText(frame, f"Общая площадь: {round(full_area_frame,2)}", (10, 50), 25  , (255,255,255))
        gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', frame)
        cv2.imshow('gray', gray_scale)

        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if time.monotonic() - save_time >= save_interval:
            save_time = time.monotonic()
            result_vid.write(frame)

            img_pil.save(f"images/frame_{counter}.jpg", "JPEG", dpi=(300, 300))

    while True:
        inRgb = qRgb.get()
        inDet = qDet.get()

        if inRgb is not None:
            frame = inRgb.getCvFrame()

            out.write(frame)  # Write frame to video file

            # Save frame as JPEG image every 0.5 seconds
            if time.monotonic() - save_time >= save_interval:
                save_time = time.monotonic()

        if inDet is not None:
            detections = inDet.detections
            counter += 1

        if frame is not None:
            displayFrame("rgb", frame, detections, save_time)

        if cv2.waitKey(1) == ord('q'):
            break

out.release()
cv2.destroyAllWindows()
