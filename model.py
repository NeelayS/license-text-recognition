import cv2
import numpy as np
import os
import shutil
import subprocess

from paddleocr import PaddleOCR
from pytorchyolo import detect, models as yolo_models


from transform import four_point_transform


class YoloV3DetectionModel:
    def __init__(self, config_path, weights_path, threshold=0.5):

        self.threshold = threshold

        self.model = yolo_models.load_model(config_path, weights_path)

    def _filter_detections(self, detections):

        relevant_class_ids = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        highest_confidence = 0.0
        filtered_detection = None

        for (
            detection
        ) in (
            detections
        ):  # To-do: Instead of filtering by confidence, filter by area of bounding box (larger the better)

            # if (
            #     detection[-2] > self.threshold and int(detection[-1]) == 2
            # ):  # 2 = vehicle class
            #     filtered_detection = list(map(lambda x: max(int(x), 0), detection[:-2]))
            #     filtered_detections.append(filtered_detection)

            if (
                detection[-2] > highest_confidence
                and int(detection[-1]) in relevant_class_ids
            ):
                filtered_detection = list(map(lambda x: max(int(x), 0), detection[:-2]))
                highest_confidence = detection[-2]

        return filtered_detection

    def __call__(self, img_path, save_path, img=None, return_detection=False):

        if img is None:
            img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        detections = detect.detect_image(self.model, img)
        detection = self._filter_detections(detections)

        if detection is None:
            raise Exception("No vehicles detected")

        img = img[detection[1] : detection[3], detection[0] : detection[2], :]
        cv2.imwrite(save_path, img)

        if return_detection:
            return detection


class TextRecognitionModel:
    def __init__(self):

        self.ocr_model = PaddleOCR(use_angle_cls=True, lang="en")

    def __call__(self, img_path):

        results = self.ocr_model.ocr(img_path, cls=True)
        texts = [result[-1][0] for result in results]

        return texts


class LicenseTextDetector:
    def __init__(
        self,
        vehicle_detection_cfg,
        vehicle_detection_weights,
        vehicle_detection_threshold,
        license_detection_weights,
    ):

        self.license_detection_weights = license_detection_weights

        self.vehicle_detection_model = YoloV3DetectionModel(
            vehicle_detection_cfg,
            vehicle_detection_weights,
            vehicle_detection_threshold,
        )
        self.text_recognition_model = TextRecognitionModel()

        self.tmp_dir = "./tmp"
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
            print("\nCreated tmp directory")

        self.vehicle_detection_img_path = os.path.join(
            self.tmp_dir, "vehicle_detection.jpg"
        )
        self.rotated_img_path = os.path.join(self.tmp_dir, "rotated_detection.jpg")
        self.resized_rotated_img_path = os.path.join(
            self.tmp_dir, "resized_rotated_detection.jpg"
        )

    def _rotate_resize_plate_detection(self, img_path, coordinates):

        img = cv2.imread(img_path)

        rotated_img = four_point_transform(img, np.array(coordinates))
        y, x, _ = np.shape(rotated_img)
        resized_rotated_img = cv2.resize(rotated_img, (x * 10, y * 10))

        cv2.imwrite(self.rotated_img_path, resized_rotated_img)
        cv2.imwrite(self.resized_rotated_img_path, resized_rotated_img)

    def _detect_license_plate(self, img_path):

        # command_str = f"python EAST/eval.py --test_data_path={img_path} --checkpoint_path={self.license_detection_weights} --output_dir={self.tmp_dir}"
        # os.system(command_str)

        subprocess.run(
            [
                "python",
                "EAST/eval.py",
                f"--test_data_path={img_path}",
                f"--checkpoint_path={self.license_detection_weights}",
                f"--output_dir={self.tmp_dir}",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        img_name = os.path.basename(img_path).split(".")[0]
        det_results_path = os.path.join(self.tmp_dir, f"{img_name}.txt")

        if not os.path.exists(det_results_path):
            raise Exception("No license plate detected")

        with open(det_results_path) as f:
            first_line = f.readline()
        f.close()

        box = first_line.split(",")
        box[7] = box[7].rstrip("\n")
        box = [int(st) for st in box]

        coordinates = [
            [box[0], box[1]],
            [box[2], box[3]],
            [box[4], box[5]],
            [box[6], box[7]],
        ]

        return coordinates

    def __call__(self, img_path=None, img=None):

        assert (
            img_path is not None or img is not None
        ), "Either img_path or CV2 img should be provided"

        self.vehicle_detection_model(
            img_path=img_path, save_path=self.vehicle_detection_img_path, img=img
        )
        print("\nDetected vehicle")

        coordinates = self._detect_license_plate(self.vehicle_detection_img_path)
        self._rotate_resize_plate_detection(
            self.vehicle_detection_img_path, coordinates
        )
        print("Detected license plate\n")

        text = self.text_recognition_model(self.resized_rotated_img_path)
        print("\nRecognized license plate text:")

        shutil.rmtree(self.tmp_dir)

        return text
