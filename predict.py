import cv2
import os
from time import time

from model import LicenseTextDetector


vehicle_detection_cfg = "configs/yolov3.cfg"
vehicle_detection_weights = "weights/yolov3.weights"
vehicle_detection_threshold = 0.05
license_detection_weights = "weights/east_vgg16.pth"
license_detection_backbone_weights = "weights/vgg16_bn-6c64b313.pth"

detector = LicenseTextDetector(
    vehicle_detection_cfg,
    vehicle_detection_weights,
    vehicle_detection_threshold,
    license_detection_weights,
    license_detection_backbone_weights=license_detection_backbone_weights,
    vehicle_detection_filter_type="area",
)


img_dir = "../data/frames/ex1"
total_time = 0

for img_path in sorted(os.listdir(img_dir)):

    img_path = os.path.join(img_dir, img_path)
    img = cv2.imread(img_path)

    start = time()
    text = detector(img=img)
    end = time()
    total_time += end - start

    print(f"Img: {img_path}, Text: {text}\n")

avg_time = total_time / len(os.listdir(img_dir))
print(f"Avg time per frame: {avg_time}")
print(f"FPS: {1 / avg_time}")
