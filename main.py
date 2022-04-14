from time import time

from model import LicenseTextDetector


def main(
    img_path,
    vehicle_detection_cfg,
    vehicle_detection_weights,
    vehicle_detection_threshold,
    license_detection_weights,
):

    detector = LicenseTextDetector(
        vehicle_detection_cfg,
        vehicle_detection_weights,
        vehicle_detection_threshold,
        license_detection_weights,
    )
    text = detector(img_path)
    print(text)


if __name__ == "__main__":

    img_path = "data/ex1.png"
    vehicle_detection_cfg = "configs/yolov3.cfg"
    vehicle_detection_weights = "weights/yolov3.weights"
    vehicle_detection_threshold = 0.25
    license_detection_weights = "weights/east_icdar2015_resnet_v1_50_rbox"

    start = time()

    main(
        img_path,
        vehicle_detection_cfg,
        vehicle_detection_weights,
        vehicle_detection_threshold,
        license_detection_weights,
    )

    end = time()
    print(f"Time taken: {end - start}")
