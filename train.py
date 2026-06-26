from ultralytics import YOLO

#load pretrained Yolo
model = YOLO("yolov8n.pt")

results = model.train(
    data = "data.yaml",
    epochs = 50,
    imgsz =640,
    batch = 16,
    patience = 10,
    project ="runs",
    name = "yolo_experiment",
    lr0 = 0.001,
    weight_decay = 0.0005
)

print("Training complete")
