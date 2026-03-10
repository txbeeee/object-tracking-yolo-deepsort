import random
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import numpy as np

model = YOLO("yolov8n.pt")
model.train(
    data="/workspace/object-tracking/data/yolo_dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    workers=0, 
    project="/workspace/object-tracking/results",
    name="yolov8_detrac"
)

#training verification
MODEL_PATH = "/workspace/object-tracking/results/yolov8_detrac2/weights/last.pt"
VAL_IMAGES = Path("/workspace/object-tracking/data/yolo_dataset/images/val")
model = YOLO(MODEL_PATH)
all_images = list(VAL_IMAGES.glob("*.jpg"))
sample_images = random.sample(all_images, 10)

print(f"selected images {len(sample_images)} ")

plt.figure(figsize=(16, 20))

for i, img_path in enumerate(sample_images, 1):
    results = model(img_path)[0]
    annotated = results.plot()
    if annotated is None:
        print(f"⚠YOLO did not return an image for {img_path}")
        continue

    plt.subplot(5, 2, i)
    plt.imshow(annotated)
    plt.title(img_path.name)
    plt.axis("off")

plt.tight_layout()
plt.show()
