from ultralytics import YOLO

#load pretrained Yolo
model = YOLO("yolov8n.pt")

results = model.train(
    data = "data.yaml",
    epochs = 50,
    imgsze = 640,
    batch = 16,
    patience = 10,
    project = "runs",
    name = "yolo_experiment"
)

print("Training complete")
