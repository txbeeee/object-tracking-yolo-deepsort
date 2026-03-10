# object-tracking-yolo-deepsort
This project implements a full object detection and tracking pipeline using:
- **YOLOv8** for vehicle detection  
- **DeepSORT** for multi-object tracking  
- **UA-DETRAC** dataset (converted to YOLO format)
## Features
- Custom-trained YOLOv8 model on DETRAC
- Minimal DeepSORT implementation (Kalman Filter + IOU matching)
- Tracking on real DETRAC video sequences
Results:
<img width="1131" height="678" alt="image" src="https://github.com/user-attachments/assets/ece217ab-f06c-4e8d-8ee8-0bf97ebe5253" />
<img width="1180" height="678" alt="image" src="https://github.com/user-attachments/assets/776353f9-525e-488d-9394-546330232236" />
