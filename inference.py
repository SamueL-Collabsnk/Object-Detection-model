from ultralytics import YOLO

model = YOLO(
    "runs/detetct/yolo_experiment/weights/best.pt"
)

results = model.predict(
    source = "test.jpg",
    conf = 0.25,
    save = True
)

print(results)