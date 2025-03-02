from ultralytics import YOLO

# model=YOLO('yolov8x.yaml').load('/home/data2/mxl/ultralytics/ultralytics-main/yolov8x.pt')
model=YOLO('/home/data2/mxl/ultralytics/ultralytics-main/runs/detect/carboard128_5/weights/last.pt')
model.val(data='/home/data2/mxl/ultralytics/ultralytics-main/ultralytics/datasets/mydatacarbox.yaml',task='detect',imgsz=128,workers=8,batch=32,patience=300,device=[1])

# model.train(data='/home/data2/mxl/ultralytics/ultralytics-main/ultralytics/datasets/mydata.yaml',epochs=300,imgsz=640,workers=8,batch=32,patience=200,device=[0,1],close_mosaic=15,resume=True)
