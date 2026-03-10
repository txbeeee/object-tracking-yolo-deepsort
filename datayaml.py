from pathlib import Path
import yaml

OUT_PATH = Path("/workspace/object-tracking/data/yolo_dataset/data.yaml")

data = {
    "train": "/workspace/object-tracking/data/yolo_dataset/images/train",
    "val": "/workspace/object-tracking/data/yolo_dataset/images/val",
    "nc": 1,
    "names": ["car"]
}

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(OUT_PATH, "w") as f:
    yaml.dump(data, f, default_flow_style=False)

print(f"data.yaml created: {OUT_PATH}")
