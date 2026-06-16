from flask import Flask
from flask import request
from flask import jsonify
from werkzeug.utils import secure_filename

from ultralytics import YOLO

import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok= True
)

model = YOLO("best.pt")

@app.route("/")
def home():
    return "YOLO API running"

@app.route(
    "/predict",
    methods= ["POST"]
)
def predict():
    if "image" not in request.files:
        return jsonify({"error":"No image uploaded"} ), 400
        
    image = request.files["image"]   
    
    if image.filename == "":
        return jsonify({"error":"Empty file"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, image.filename)

    image.save(filepath)

    results = predict(source = filepath,conf = 0.25)

    detections = []

    for result in results:
      boxes = result.boxes
    
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        
        detections.append(
            {
                "class": model.names[cls],
                "confidence": round(conf, 4)
            }
        )    
    return jsonify(
            {
              "detections": detections
            }
    )   
        
if __name__ == "__main__":
    app.run(
        host ="0.0.0.0",
        port = 5000,
        debug = True
    )
   
    


