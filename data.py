import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from sklearn.model_selection import train_test_split

DETRAC_ROOT = Path("/workspace/object-tracking/data/detrac")
IMAGES_ROOT = DETRAC_ROOT / "DETRAC-Images" / "DETRAC-Images"
TRAIN_XML = DETRAC_ROOT / "DETRAC-Train-Annotations-XML" / "DETRAC-Train-Annotations-XML"

YOLO_ROOT = Path("/workspace/object-tracking/data/yolo_dataset")
YOLO_IMAGES = YOLO_ROOT / "images"
YOLO_LABELS = YOLO_ROOT / "labels"

for p in [YOLO_IMAGES, YOLO_LABELS]:
    (p / "train").mkdir(parents=True, exist_ok=True)
    (p / "val").mkdir(parents=True, exist_ok=True)

def convert_bbox_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    x_center = (xmin + xmax) / 2.0 / img_w
    y_center = (ymin + ymax) / 2.0 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return x_center, y_center, w, h

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    frames = root.findall("frame")
    annotations = {}

    for frame in frames:
        frame_id = int(frame.attrib["num"])
        targets = frame.find("target_list").findall("target")
        boxes = []
        for t in targets:
            box = t.find("box").attrib
            xmin = float(box["left"])
            ymin = float(box["top"])
            xmax = xmin + float(box["width"])
            ymax = ymin + float(box["height"])
            boxes.append((xmin, ymin, xmax, ymax))
        annotations[frame_id] = boxes
    return annotations
  
videos = sorted([d for d in IMAGES_ROOT.iterdir() if d.is_dir()])
print(f"videos found: {len(videos)}")

train_videos, val_videos = train_test_split(videos, test_size=0.2, random_state=42)

print(f"Train videos: {len(train_videos)}")
print(f"Val videos: {len(val_videos)}")


def process_video(video_dir, xml_dir, split):
    video_name = video_dir.name
    xml_path = xml_dir / f"{video_name}.xml"

    if not xml_path.exists():
        print(f"[WARN] No XML for {video_name}")
        return

    print(f"[INFO] Processing {video_name}")
    annotations = parse_xml(xml_path)

    for img_file in sorted(video_dir.glob("*.jpg")):
        frame_id = int(img_file.stem.replace("img", ""))
        if frame_id not in annotations:
            continue

        out_img = YOLO_IMAGES / split / img_file.name
        shutil.copy(img_file, out_img)

        label_path = YOLO_LABELS / split / (img_file.stem + ".txt")
      
        import cv2
        img = cv2.imread(str(img_file))
        h, w = img.shape[:2]

        with open(label_path, "w") as f:
            for (xmin, ymin, xmax, ymax) in annotations[frame_id]:
                xc, yc, bw, bh = convert_bbox_to_yolo(xmin, ymin, xmax, ymax, w, h)
                f.write(f"0 {xc} {yc} {bw} {bh}\n")

for v in train_videos:
    process_video(v, TRAIN_XML, "train")

for v in val_videos:
    process_video(v, TRAIN_XML, "val")

print("The YOLO dataset has been created")
