from ultralytics import YOLO

model = YOLO('yolo11n.pt')

results = model.train(data="./dataset/data.yaml", epochs=80, imgsz=640, patience=15, batch=32)

