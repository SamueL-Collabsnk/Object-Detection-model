from ultralytics import YOLO

# Load trained model (correct path)
model = YOLO("runs/detect/runs/yolo_experiment-3/weights/best.pt")

# Run inference
results = model.predict(
    source="test/images",
    save=True,
    show=True,
    conf=0.25
)

print("Inference complete")






