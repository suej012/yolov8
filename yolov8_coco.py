
from ultralytics import YOLO

import os
import cv2
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # 또는 Qt5Agg, GTK3Agg 등

def show_results(model, image_folder):
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    max_images = min(len(image_files), 16)  # 예: 최대 16장까지만 표시
    cols = 4
    rows = (max_images + cols - 1) // cols

    plt.figure(figsize=(15, 5 * rows))
    for i in range(max_images):
        image_path = os.path.join(image_folder, image_files[i])
        
        results = model(image_path)
        
        img = results[0].plot()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img_rgb)
        plt.title(image_files[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def test_coco8():
    model = YOLO("./models/yolov8m.pt")

    results = model.train(data="coco8.yaml", epochs=10)
    results = model.val()

    show_results(model, "./datasets/coco8/images/train")
    # show_results(model, "./datasets/coco8/images/val")

def test_coco():
    model = YOLO("./models/yolov8n.pt")

    results = model.train(data="coco.yaml", epochs=1)
    
    show_results(model, "./datasets/coco/images/test2017")


if __name__ == "__main__":
    test_coco8()
    # test_coco()
    