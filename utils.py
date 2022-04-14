import cv2
from torchvision import transforms


def resize_img(img):
    """resize image to be divisible by 32"""
    h, w, _ = img.shape
    resize_w = w
    resize_h = h

    resize_h = resize_h if resize_h % 32 == 0 else int(resize_h / 32) * 32
    resize_w = resize_w if resize_w % 32 == 0 else int(resize_w / 32) * 32
    img = cv2.resize(img, (resize_w, resize_h), cv2.INTER_LINEAR)
    ratio_h = resize_h / h
    ratio_w = resize_w / w

    return img, ratio_h, ratio_w


def cv2_to_tensor(img):
    """convert cv2 image to torch.Tensor"""
    t = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    return t(img).unsqueeze(0)
