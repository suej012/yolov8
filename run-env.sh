#!/bin/bash

docker stop yolov8-env || true
docker rm yolov8-env || true

docker run --gpus all -it \
  -v ~/Workspace/yolov8:/workspace \
  --shm-size=32g \
  --ipc=host \
  --network=host \
  --name yolov8-env \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  yolov8-env:latest

