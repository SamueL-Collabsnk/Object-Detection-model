from ultralytics import YOLO

model = YOLO(
    "runs/detect/yolo_experiment/weights/best.pt"
    )

metrics = model.val()

print(metrics)