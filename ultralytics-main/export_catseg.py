from ultralytics import YOLO

model=YOLO('yolov8n-seg.yaml').load('/home/data2/mxl/ultralytics/ultralytics-main/runs/segment/train1207/weights/last.pt')
path = model.export(format="onnx",opset=12,simplify=True,imgsz=640)  # export the model to ONNX format

