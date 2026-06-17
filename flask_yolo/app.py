from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

MODEL_PATH = (
    "runs/detect/runs/yolo_experiment-3/weights/best.pt"
)
model = None

if os.path.isfile(MODEL_PATH):
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Failed to load model at {MODEL_PATH}:{e}")  
        model = None  
else:
    print(f"Model file not found at {MODEL_PATH}")#App will run but detect will return error untill model is provided        


UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error":"No file part 'image' in request"}),400

    file = request.files["image"]
    
    if file.filename == "":
        return jsonify({"error":"Empty filename"}),400
    filename = secure_filename(file.filename)

    path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    file.save(path)

    if model is None:
        return jsonify({"error": "Model not loaded.Place weights at MODEL_PATH"}) 
    try:
        results = model.predict(source=path, conf = 0.25)
    except Exception as e:
        return jsonify({"error":"Prediction failed"}),500    
         
    detections = []

    for r in results:
        for box in getattr(r, "boxes", []):
            detections.append({
                "class": int(box.cls[0])if hasattr (box, "cls") else None,
                "confidence": float(box.conf[0]) if hasattr (box, "conf") else None
            })

    return jsonify(detections)


if __name__ == "__main__":
    app.run(debug=True)

