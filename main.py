from time import time

from model import LicenseTextDetector


def main(
    img_path,
    vehicle_detection_cfg,
    vehicle_detection_weights,
    vehicle_detection_threshold,
    license_detection_weights,
    license_detection_backbone_weights=None,
    use_east_tf=False,
):

    detector = LicenseTextDetector(
        vehicle_detection_cfg,
        vehicle_detection_weights,
        vehicle_detection_threshold,
        license_detection_weights,
        license_detection_backbone_weights=license_detection_backbone_weights,
        use_east_tf=use_east_tf,
    )

    start = time()
    text = detector(img_path)
    end = time()

    print(text)
    print(f"Time taken: {end - start}")


if __name__ == "__main__":

    img_path = "data/ex5.png"
    vehicle_detection_cfg = "configs/yolov3.cfg"
    vehicle_detection_weights = "weights/yolov3.weights"
    vehicle_detection_threshold = 0.25
    # license_detection_weights = "weights/east_icdar2015_resnet_v1_50_rbox"  # EAST TF model weights
    license_detection_weights = "weights/east_vgg16.pth"
    license_detection_backbone_weights = "weights/vgg16_bn-6c64b313.pth"

    main(
        img_path,
        vehicle_detection_cfg,
        vehicle_detection_weights,
        vehicle_detection_threshold,
        license_detection_weights,
        license_detection_backbone_weights=license_detection_backbone_weights,
        use_east_tf=False,
    )
