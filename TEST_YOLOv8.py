import cv2
from ultralytics import YOLO
import matplotlib
import matplotlib.pyplot as plt

# Create a new YOLO model from scratch
# model = YOLO("yolo11n.yaml")

# Load a pretrained YOLO model (recommended for training)
model = YOLO("./models/yolov8n.pt")

# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(data="coco8.yaml", epochs=3)

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
results = model("https://ultralytics.com/images/bus.jpg")
img = results[0].plot()

if True:
    cv2.imshow("YOLOv8", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    matplotlib.use('TkAgg')  # 또는 Qt5Agg, GTK3Agg 등
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

# Export the model to ONNX format
success = model.export(format="onnx")