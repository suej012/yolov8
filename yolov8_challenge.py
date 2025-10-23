import cv2
from ultralytics import YOLO

model = YOLO("./models/yolov8n.pt")

video_path = "./datasets/challenge/dongdaegu4.avi"
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
target_fps = 10
frame_interval = int(fps/target_fps)
print(f"fps: {fps}, target_fps: {target_fps}, frame_interval: {frame_interval}")

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        results = model(frame)
        annotated_frame = results[0].plot()
    else:
        # 이전 결과 그대로, tracking 할수도.
        pass

    frame_count += 1

    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # ESC 키 또는 'q' 누르면 종료
    if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
        break

cap.release()
cv2.destroyAllWindows()
