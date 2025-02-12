import cv2
import numpy as np
import torch
from ultralytics import YOLO

def detect_fire(image_path):
    model = YOLO("../models/yolov8-wildfire.pt")  # Pretrained Model
    image = cv2.imread(image_path)
    results = model(image)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Bounding Box
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    cv2.imwrite("../static/detected_fire.jpg", image)
