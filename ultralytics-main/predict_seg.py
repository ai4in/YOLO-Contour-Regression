from ultralytics import YOLO

model=YOLO('/home/data2/mxl/ultralytics/ultralytics-main/runs/segment/addcyclegan_img/weights/last.pt')
for r in model.predict(source="/home/data2/mxl/ultralytics/ultralytics-main/dataset/onlytree/images/val",save=True,conf=0.5,stream=True,imgsz=640):
    pass