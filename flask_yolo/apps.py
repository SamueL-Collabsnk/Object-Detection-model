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
MODEL_PATH = "best.pt"
model = None

if os.path.isfile(MODEL_PATH):
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Failed t load {MODEL_PATH}:{e}") 
        model = None
    else:
        print(f"{MODEL_PATH} not found. Place weights at{MODEL_PATH} or change MODEL_PATH.")
               
        

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
    if model is None:
        return jsonify ({
            "error":"Model not loaded. Put best.pt  at project root or update MODEL_PATH."
        })
        
    image = request.files["image"]   
    
    if image.filename == "":
        return jsonify({"error":"Empty file"}), 400

    filename = secure_filename(image.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    image.save(filepath)

    results = predict(source = filepath,conf = 0.25)
    results = model.predict(source=filepath, conf=0.25)

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
    for result in results:
        boxex = getattr(result, "boxes", [])
        for box in boxes:
            try:
                cls = int(box.cls[0])
                conf = float(box.conf[0])    
            except Exception:
                 continue
            label = model.names[cls] if  isinstance(
                 model.names, (list,tuple)) else model.names.get(cls, str(cls))
            detections.append({"class": label, "confidence": round(conf, 4)})
                
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
   
    


