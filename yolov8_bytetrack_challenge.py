import cv2
from ultralytics import YOLO
import numpy as np
from norfair import Detection, Tracker
import time

# 추적 대상 클래스 (YOLOv8의 COCO 클래스 기준)
# 0 = person, 1 = bicycle, 2 = car, 3 = motorcycle, 5 = bus, 7 = truck
TARGET_CLASSES = {0, 1, 2, 3, 5, 7}

# model = YOLO("yolov8n.pt")
model = YOLO("./models/yolov8m.pt")
tracker = Tracker(distance_function="euclidean", distance_threshold=70, hit_counter_max=80, initialization_delay=0)

def yolo_to_norfair_detections(yolo_result):
    """YOLOv8 결과를 Norfair Detection 형태로 변환"""
    detections = []
    boxes = yolo_result.boxes.xyxy.cpu().numpy()
    confs = yolo_result.boxes.conf.cpu().numpy()
    clss = yolo_result.boxes.cls.cpu().numpy().astype(int)

    for box, conf, cls in zip(boxes, confs, clss):
        if cls not in TARGET_CLASSES:
            continue

        x1, y1, x2, y2 = box
        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2
        detections.append(Detection(points=np.array([[xc, yc]]), scores=np.array([conf]), label=str(cls)))

    return detections

video_path = "./datasets/challenge/younam4.avi"
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)   # 200 - 비정상적 값.
if fps <= 0 or fps > 120:
    fps = 30

target_fps = 10
frame_interval = int(fps/target_fps)
print(f"fps: {fps}, target_fps: {target_fps}, frame_interval: {frame_interval}")

frame_count = 0
while cap.isOpened():
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        results = model(frame, imgsz=1280, conf=0.2, iou=0.45)
        # annotated_frame = results[0].plot()
        detections = yolo_to_norfair_detections(results[0])
        tracked_objects = tracker.update(detections)
    else:
        tracked_objects = tracker.update(detections)

    frame_count += 1

    # print("Tracked objects:", tracked_objects)
    for tracked in tracked_objects:
        x, y = tracked.estimate[0]
        track_id = tracked.id
        class_id = tracked.last_detection.label if tracked.last_detection else "?"
        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
        cv2.putText(frame, f"ID: {track_id} ({class_id})", (int(x), int(y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # cv2.imshow("YOLOv8 Detection", annotated_frame)
    cv2.imshow("YOLOv8 + ByteTrack", frame)

    elapsed = time.time() - start_time
    delay = int(1000/fps)
    delay = max(int((1000/fps - elapsed*1000)), 1)  # ms 단위, 최소 1ms

    # ESC 키 또는 'q' 누르면 종료
    if cv2.waitKey(delay) & 0xFF in [27, ord('q')]:
        break

cap.release()
cv2.destroyAllWindows()
