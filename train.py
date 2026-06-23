from ultralytics import YOLO

#load pretrained Yolo
model = YOLO("yolov8n.pt")

results = model.train(
    data = "data.yaml",
    epochs = 50,
    imgsz = 768,
    batch = 32,
    patience = 10,
    project ="runs",
    name = "yolo_experiment",
    lr0 = 0.0001,
    weight_decay = 0.0005
)

print("Training complete")
