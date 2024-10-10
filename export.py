from ultralytics import YOLO

model = YOLO("runs/detect/train_l/weights/best.pt")

model.export(format="onnx")