from ultralytics import YOLO

model=YOLO('yolov8n-cls.yaml').load('/home/data2/mxl/ultralytics/ultralytics-main/runs/classify/train54/weights/best.pt')
path = model.export(format="onnx",opset=12,simplify=True,imgsz=224)  # export the model to ONNX format

