import cv2
from ultralytics import YOLO
from pathlib import Path

model = YOLO("/workspace/object-tracking/results/yolov8_detrac2/weights/best.pt")
tracker = Tracker()

VIDEO_PATH = "/workspace/object-tracking/data/videos/test.mp4"
OUT_PATH = Path("/workspace/object-tracking/results/tracking/output.mp4")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ Failed to open video:", VIDEO_PATH)
    exit()

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(str(OUT_PATH), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

if not out.isOpened():
    print("❌ VideoWriter did not open the file:", OUT_PATH)
    exit()
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    if frame_idx % 30 == 0:
        print(f"Frames processed: {frame_idx}")

    results = model(frame)[0]
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        detections.append(Detection([x1, y1, x2 - x1, y2 - y1], conf))

    tracker.update(detections)
    for track in tracker.tracks:
        x, y, w, h = track.mean[:4]
        cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0,255,0), 2)
        cv2.putText(frame, f"ID {track.track_id}", (int(x), int(y)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    out.write(frame)
cap.release()
out.release()

print("✅ The video is saved in:", OUT_PATH)
