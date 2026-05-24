"""
Bone Cancer Detection — Flask Backend
Uses YOLOv8 classification model (best.pt) to classify X-ray images
as 'cancer' or 'normal'.

Directory layout expected:
  bone_cancer_app/
  ├── app.py
  ├── templates/
  │   └── index.html
  └── weights/
      └── best.pt          ← your trained YOLOv8 weights

Run:
  pip install flask ultralytics pillow
  python app.py
"""

import io
import os
import base64
import traceback

from flask import Flask, request, jsonify, render_template
from PIL import Image
from ultralytics import YOLO

# ── configuration ────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "weights/best.pt")
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

# ── app setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── load model once at startup ────────────────────────────────────────────────
print(f"[INFO] Loading model from: {MODEL_PATH}")
try:
    model = YOLO(MODEL_PATH, task="classify")
    print("[INFO] Model loaded ✅")
except Exception as e:
    print(f"[WARN] Could not load model: {e}")
    model = None


# ── helpers ──────────────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Use JPG, PNG, or BMP."}), 400
    if model is None:
        return jsonify({"error": "Model not loaded. Place best.pt in ./weights/"}), 503

    try:
        # Read and convert image
        img_bytes = file.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Run inference
        results = model(pil_img, verbose=False)
        r = results[0]

        # Extract predictions
        probs = r.probs
        top1_idx = probs.top1
        top1_label = r.names[top1_idx]
        top1_conf = float(probs.top1conf)

        # Build full class probability table
        class_probs = {
            r.names[i]: round(float(probs.data[i]), 4)
            for i in range(len(r.names))
        }

        # Encode image for response (thumbnail)
        thumb = pil_img.copy()
        thumb.thumbnail((512, 512))
        img_b64 = image_to_base64(thumb)

        return jsonify({
            "prediction": top1_label,
            "confidence": round(top1_conf * 100, 2),
            "class_probabilities": class_probs,
            "image_b64": img_b64,
            "status": "success",
        })

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Inference failed. Check server logs."}), 500


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
    })


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)