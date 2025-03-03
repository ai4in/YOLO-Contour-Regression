from ultralytics import YOLO

model=YOLO('yolov8n-seg.yaml')
model.train(data='/home/data2/mxl/ultralytics/ultralytics-main/ultralytics/datasets/bdd100k.yaml',epochs=300,task='segment',mixup=0,mosaic=1,imgsz=640,workers=2,batch=32,patience=50,device=0)
