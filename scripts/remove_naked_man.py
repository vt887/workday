import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove the rightmost person from an image using segmentation + inpainting")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", required=True, help="Path to save the edited image")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold for detection (default: 0.3)")
    parser.add_argument("--dilate", type=int, default=9, help="Kernel size for mask dilation (odd number, default: 9)")
    parser.add_argument("--model", default="yolov8s-seg.pt", help="YOLOv8 segmentation model to use (default: yolov8s-seg.pt)")
    return parser.parse_args()


def ensure_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def remove_rightmost_person(input_path: str, output_path: str, conf: float, dilate_kernel: int, model_name: str) -> None:
    input_path_obj = Path(input_path)
    if not input_path_obj.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    image_bgr = cv2.imread(str(input_path_obj), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {input_path}")

    model = YOLO(model_name)
    results = model.predict(source=image_bgr, verbose=False, imgsz=1024, conf=conf, classes=[0])  # class 0 = person
    result = results[0]

    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        # No person detected; just save original
        cv2.imwrite(str(output_path), image_bgr)
        return

    boxes_xyxy = result.boxes.xyxy.cpu().numpy()  # (n,4)
    masks_data = result.masks.data.cpu().numpy().astype(np.uint8)  # (n,h,w) values 0/1

    # Pick the person whose bounding-box center-x is the largest (rightmost)
    centers_x = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.0
    rightmost_index = int(np.argmax(centers_x))

    person_mask = masks_data[rightmost_index] * 255  # to 0/255

    # Optionally dilate the mask to remove halos
    ksize = ensure_odd(max(3, dilate_kernel))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated_mask = cv2.dilate(person_mask, kernel, iterations=1)

    # OpenCV inpainting expects 8-bit single-channel mask, non-zero areas are inpainted
    inpaint_radius = max(3, ksize // 3)
    inpainted = cv2.inpaint(image_bgr, dilated_mask, inpaint_radius, cv2.INPAINT_TELEA)

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path_obj), inpainted)


def main() -> None:
    args = parse_args()
    remove_rightmost_person(
        input_path=args.input,
        output_path=args.output,
        conf=args.conf,
        dilate_kernel=args.dilate,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()

