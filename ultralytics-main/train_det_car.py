from ultralytics import YOLO

# model=YOLO('yolov8n.yaml').load('/home/data2/mxl/ultralytics/ultralytics-main/yolov8n.pt')
model=YOLO('yolov8n.yaml').load('/home/data2/mxl/ultralytics/ultralytics-main/runs/detect/carboard128_4/weights/last.pt')

# model=YOLO('/home/data2/mxl/ultralytics/ultralytics-main/runs/detect/train4/weights/last.pt')
# model=YOLO('/home/data2/mxl/ultralytics/ultralytics-main/runs/segment/train51/weights/best.pt')
# model.train(data='/home/data2/mxl/ultralytics/ultralytics-main/ultralytics/datasets/mydata.yaml',mixup=0,
# mosaic=0,epochs=300,imgsz=640,workers=8,batch=32,patience=300,device=[1,3])
model.train(data='/home/data2/mxl/ultralytics/ultralytics-main/ultralytics/datasets/mydatacarbox.yaml',task='detect',epochs=300,mixup=0,mosaic=0,imgsz=128,workers=8,batch=256,patience=30,device=[3])
