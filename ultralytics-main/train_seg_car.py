from ultralytics import YOLO

model=YOLO('yolov8n-seg.yaml')
# model=YOLO('yolov8n-seg.yaml').load('/home/data2/mxl/ultralytics/ultralytics-main/yolov8n-seg.pt')

# model=YOLO('/home/data2/mxl/ultralytics/ultralytics-main/runs/detect/train4/weights/last.pt')
# model=YOLO('/home/data2/mxl/ultralytics/ultralytics-main/runs/segment/train51/weights/best.pt')
# model.train(data='/home/data2/mxl/ultralytics/ultralytics-main/ultralytics/datasets/mydata.yaml',mixup=0,
# mosaic=0,epochs=300,imgsz=640,workers=8,batch=32,patience=300,device=[1,3])
model.train(data='/home/data2/mxl/ultralytics/ultralytics-main/ultralytics/datasets/bdd100k.yaml',epochs=300,task='segment',mixup=0,mosaic=1,imgsz=640,workers=8,batch=32,patience=50,device=[1])
