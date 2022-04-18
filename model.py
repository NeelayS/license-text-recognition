import cv2
import numpy as np
import os
import shutil
import subprocess
import torch

from paddleocr import PaddleOCR
from pytorchyolo import detect, models as yolo_models

from EAST_torch.detect import get_boxes, adjust_ratio
from EAST_torch.models import EAST
from transform import four_point_transform
from utils import resize_img, cv2_to_tensor


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

    def __call__(
        self,
        img_path,
        save_path,
        img=None,
        return_detection=False,
        save_detected_img=False,
    ):

        if img is None:
            img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        detections = detect.detect_image(self.model, img)
        detection = self._filter_detections(detections)

        if detection is None:
            print("No vehicles detected")
            return None

        img = img[detection[1] : detection[3], detection[0] : detection[2], :]

        if save_detected_img:
            cv2.imwrite(save_path, img)

        if return_detection:
            return detection

        return img


class TextRecognitionModel:
    def __init__(self):

        self.ocr_model = PaddleOCR(use_angle_cls=True, lang="en")

    def __call__(self, img=None):

        results = self.ocr_model.ocr(img, cls=True)
        if len(results) == 0 or results is None:
            print("No text detected")
            return None

        texts = [result[-1][0] for result in results]

        return texts


class EASTDetectionModel:
    def __init__(self, detector_weights_path, backbone_weights_path):

        self.model = EAST(pretrained_path=backbone_weights_path)
        self.model.load_state_dict(
            torch.load(detector_weights_path, map_location=torch.device("cpu"))
        )
        self.model.eval()

    def __call__(self, img):

        img, ratio_h, ratio_w = resize_img(img)
        img = cv2_to_tensor(img)

        with torch.no_grad():
            score, geo = self.model(img)

        boxes = get_boxes(score.squeeze(0).cpu().numpy(), geo.squeeze(0).cpu().numpy())

        if boxes is None:
            boxes = []

        if len(boxes) == 0:
            print("No vehicles detected")
            return None

        return adjust_ratio(boxes, ratio_w, ratio_h)[0]


class LicenseTextDetector:
    def __init__(
        self,
        vehicle_detection_cfg,
        vehicle_detection_weights,
        vehicle_detection_threshold,
        license_detection_weights,
        license_detection_backbone_weights=None,
        use_east_tf=False,
        tmp_dir=None,
    ):

        self.use_east_tf = use_east_tf
        self.tmp_dir = "./tmp" if tmp_dir is None else tmp_dir

        if use_east_tf is False:
            assert (
                license_detection_backbone_weights is not None
            ), "Backbone weights path is required"

            self.text_detection_model = EASTDetectionModel(
                detector_weights_path=license_detection_weights,
                backbone_weights_path=license_detection_backbone_weights,
            )

        else:
            self.license_detection_weights = license_detection_weights
            os.makedirs(self.tmp_dir, exist_ok=True)

        self.vehicle_detection_model = YoloV3DetectionModel(
            vehicle_detection_cfg,
            vehicle_detection_weights,
            vehicle_detection_threshold,
        )
        self.text_recognition_model = TextRecognitionModel()

        self.vehicle_detection_img_path = os.path.join(
            self.tmp_dir, "vehicle_detection.jpg"
        )
        self.rotated_img_path = os.path.join(self.tmp_dir, "rotated_detection.jpg")
        self.resized_rotated_img_path = os.path.join(
            self.tmp_dir, "resized_rotated_detection.jpg"
        )

    def _rotate_resize_plate_detection(
        self, coordinates, img=None, img_path=None, save_img=False
    ):

        assert (
            img is not None or img_path is not None
        ), "Image or image path is required"

        if img is None:
            img = cv2.imread(img_path)
            save_img = True

        rotated_img = four_point_transform(img, np.array(coordinates))
        y, x, _ = np.shape(rotated_img)
        resized_rotated_img = cv2.resize(rotated_img, (x * 10, y * 10))

        if save_img:
            cv2.imwrite(self.rotated_img_path, resized_rotated_img)
            cv2.imwrite(self.resized_rotated_img_path, resized_rotated_img)

        return resized_rotated_img

    def _run_east_tf(self, img_path):

        # command_str = f"python EAST_tf/eval.py --test_data_path={img_path} --checkpoint_path={self.license_detection_weights} --output_dir={self.tmp_dir}"
        # os.system(command_str)

        subprocess.run(
            [
                "python",
                "EAST_tf/eval.py",
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
            print("No license plate detected")
            return None

        with open(det_results_path) as f:
            first_line = f.readline()
        f.close()

        box = first_line.split(",")
        box[7] = box[7].rstrip("\n")
        box = [int(st) for st in box]

        return box

    def _detect_license_plate(self, img=None, img_path=None):

        if self.use_east_tf:
            assert img_path is not None, "Image path is required"
            box = self._run_east_tf(img_path)

        else:
            assert img is not None, "Image is required"
            box = self.text_detection_model(img)

        if box is None:
            return None

        coordinates = [
            [box[0], box[1]],
            [box[2], box[3]],
            [box[4], box[5]],
            [box[6], box[7]],
        ]

        return coordinates

    def __call__(self, img=None, img_path=None, del_tmp_dir=True):

        assert (
            img_path is not None or img is not None
        ), "Either img_path or CV2 img should be provided"

        detected_img = self.vehicle_detection_model(
            img_path=img_path, save_path=self.vehicle_detection_img_path, img=img
        )
        if detected_img is None:
            return None
        # print("\nDetected vehicle")

        coordinates = self._detect_license_plate(
            img=detected_img, img_path=self.vehicle_detection_img_path
        )
        if coordinates is None:
            return None

        rotated_resized_img = self._rotate_resize_plate_detection(
            coordinates=coordinates,
            img=detected_img,
            img_path=self.vehicle_detection_img_path,
        )
        # print("Detected license plate\n")

        text = self.text_recognition_model(img=rotated_resized_img)
        if text is None:
            return None
        # print("\nRecognized license plate text:")

        if self.use_east_tf and del_tmp_dir:
            shutil.rmtree(self.tmp_dir)

        return text
