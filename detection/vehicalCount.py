from pathlib import Path

import cv2
from ultralytics import YOLO

MODEL_PATH = "traffic_detection_yolov8s.pt"
SOURCE = "../data/test/00 (182).png"   # folder of images or video file

model = YOLO(MODEL_PATH)

def count_vehicles_in_image(img_path):
    results = model(img_path, conf=0.25, iou=0.5)[0]
    count = len(results.boxes) if results.boxes is not None else 0
    return count, results

def run_on_images(folder):
    folder = Path(folder)
    for img_path in folder.glob("*.*"):
        count, results = count_vehicles_in_image(str(img_path))
        print(f"{img_path.name}: {count} vehicles detected")

        # optional: visualize
        annotated = results.plot()
        cv2.imwrite(f"outputs/{img_path.name}", annotated)

if __name__ == "__main__":
    Path("outputs").mkdir(exist_ok=True)
    run_on_images(SOURCE)