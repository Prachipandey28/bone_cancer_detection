"""
Bone Cancer Detection — Flask Backend
Uses YOLOv8 classification model (best.pt) to classify X-ray images
as 'cancer' or 'normal'.

Directory layout expected:
  bone_cancer_app/
  ├── app.py
  ├── templates/
  │   ├── index.html
  │   └── login.html
  └── weights/
      └── best.pt          ← your trained YOLOv8 weights

Run:
  pip install flask ultralytics pillow
  python app.py
"""

import io
import os
import json
import time
import base64
import datetime
import traceback

# Prevent Intel OpenMP duplicate library initialization crash on macOS Anaconda environments
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from PIL import Image
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from ultralytics.data.augment import classify_transforms

# ── configuration ────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "weights/best.pt")
# Fallback to default weights if MODEL_PATH points to a directory or invalid path
if not MODEL_PATH or os.path.isdir(MODEL_PATH) or not os.path.exists(MODEL_PATH):
    MODEL_PATH = "weights/best.pt"
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
USERS_FILE = "users.json"

# ── app setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.secret_key = "ostscan-secure-session-key-2026"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── load model once at startup ────────────────────────────────────────────────
print(f"[INFO] Loading model from: {MODEL_PATH}")
try:
    model = YOLO(MODEL_PATH, task="classify")
    print("[INFO] Model loaded ✅")
except Exception as e:
    print(f"[WARN] Could not load model: {e}")
    model = None


# ── thread-safe Grad-CAM generator ────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        self.forward_handle = self.target_layer.register_forward_hook(self.save_activation)
        self.backward_handle = self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output.detach()
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        
    def __call__(self, x, target_class=0):
        self.model.zero_grad()
        outputs = self.model(x)
        
        if isinstance(outputs, tuple):
            logits = outputs[1]
        else:
            logits = outputs
            
        score = logits[0, target_class]
        score.backward()
        
        if self.activations is None or self.gradients is None:
            return None
            
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        weighted_activations = self.activations * pooled_gradients.view(1, self.activations.shape[1], 1, 1)
        heatmap = torch.mean(weighted_activations, dim=1).squeeze()
        heatmap = torch.clamp(heatmap, min=0)
        
        max_val = torch.max(heatmap)
        if max_val > 0:
            heatmap /= max_val
            
        return heatmap.cpu().numpy()
        
    def release(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


# ── helpers ──────────────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def load_users() -> dict:
    if not os.path.exists(USERS_FILE):
        # Create an empty file if it doesn't exist
        save_users({})
        return {}
    try:
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Could not read user database: {e}")
        return {}


def save_users(users: dict):
    try:
        with open(USERS_FILE, "w") as f:
            json.dump(users, f, indent=4)
    except Exception as e:
        print(f"[WARN] Could not write user database: {e}")


# ── routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("logged_in"):
        return redirect(url_for("index"))
        
    error = None
    if request.method == "POST":
        # Support both form data and JSON requests
        if request.is_json:
            data = request.get_json()
            username = data.get("username", "") # can be email or name
            password = data.get("password", "")
        else:
            username = request.form.get("username", "")
            password = request.form.get("password", "")
            
        username_query = username.lower().strip()
        password_query = password.strip()
        
        users = load_users()
        user_found = None
        user_email = None
        
        # Check by email key
        if username_query in users:
            user_found = users[username_query]
            user_email = username_query
        else:
            # Check by Name field (case insensitive match)
            for email_key, user_data in users.items():
                if user_data.get("name", "").lower().strip() == username_query:
                    user_found = user_data
                    user_email = email_key
                    break
                    
        if user_found and user_found.get("password") == password_query:
            session["logged_in"] = True
            session["username"] = user_found.get("name")
            session["email"] = user_email
            if request.is_json:
                return jsonify({"status": "success", "message": "Authenticated"})
            return redirect(url_for("index"))
        else:
            error = "Invalid email/name or passcode."
            if request.is_json:
                return jsonify({"status": "error", "message": error}), 401
                
    return render_template("login.html", error=error)


@app.route("/register", methods=["GET", "POST"])
def register():
    if session.get("logged_in"):
        return redirect(url_for("index"))
        
    register_error = None
    if request.method == "POST":
        # Support both form data and JSON requests
        if request.is_json:
            data = request.get_json()
            name = data.get("name", "")
            email = data.get("email", "")
            password = data.get("password", "")
        else:
            name = request.form.get("name", "")
            email = request.form.get("email", "")
            password = request.form.get("password", "")
            
        name = name.strip()
        email = email.lower().strip()
        password = password.strip()
        
        if not name or not email or not password:
            register_error = "All fields (Name, Email, Passcode) are required."
            if request.is_json:
                return jsonify({"status": "error", "message": register_error}), 400
            return render_template("login.html", register_error=register_error, active_tab="register")
            
        users = load_users()
        if email in users:
            register_error = "This email address is already registered."
            if request.is_json:
                return jsonify({"status": "error", "message": register_error}), 400
            return render_template("login.html", register_error=register_error, active_tab="register")
            
        # Register the user
        users[email] = {
            "name": name,
            "password": password
        }
        save_users(users)
        
        # Log in the user automatically
        session["logged_in"] = True
        session["username"] = name
        session["email"] = email
        
        if request.is_json:
            return jsonify({"status": "success", "message": "Account registered successfully."})
        return redirect(url_for("index"))
        
    return render_template("login.html", active_tab="register")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/history", methods=["GET"])
def history():
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized session. Please login."}), 401
        
    email = session.get("email")
    users = load_users()
    if email in users:
        records = users[email].get("history", [])
        # Return sorted by newest first (reverse chronological)
        return jsonify(records[::-1])
    return jsonify([])


@app.route("/history/notes", methods=["POST"])
def history_notes():
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized session. Please login."}), 401
        
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing payload"}), 400
        
    case_id = data.get("case_id")
    notes = data.get("notes", "").strip()
    
    if not case_id:
        return jsonify({"error": "Case ID is required"}), 400
        
    email = session.get("email")
    users = load_users()
    
    if email in users and "history" in users[email]:
        for record in users[email]["history"]:
            if record["case_id"] == case_id:
                record["notes"] = notes
                save_users(users)
                return jsonify({"status": "success", "message": "Notes saved to case history."})
                
    return jsonify({"error": "Case record not found."}), 404


@app.route("/predict", methods=["POST"])
def predict():
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized session. Please login."}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Use JPG, PNG, or BMP."}), 400
    if model is None:
        return jsonify({"error": "Model not loaded. Place best.pt in ./weights/"}), 503

    # Patient metadata values
    patient_name = request.form.get("patientName", "PAT-TEMP").strip()
    patient_age = request.form.get("patientAge", "—").strip()
    anatomical_site = request.form.get("anatomicalSite", "Other / General").strip()

    try:
        # Read and convert image
        img_bytes = file.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Save visual file to static uploads folder for persistent history
        file_ext = file.filename.rsplit(".", 1)[1].lower() if "." in file.filename else "png"
        timestamp_sec = int(time.time())
        safe_email = session.get("email", "anonymous").replace("@", "_").replace(".", "_")
        filename = f"scan_{safe_email}_{timestamp_sec}.{file_ext}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        pil_img.save(filepath, "PNG")
        image_url = f"/static/uploads/{filename}"

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

        # Generate Grad-CAM Heatmap
        heatmap_url = None
        try:
            if hasattr(model, "model"):
                py_model = model.model
                if len(py_model.model) > 9 and hasattr(py_model.model[9], "conv"):
                    target_layer = py_model.model[9].conv
                    
                    # Preprocess for PyTorch
                    transform = classify_transforms(224)
                    img_tensor = transform(pil_img).unsqueeze(0)
                    img_tensor.requires_grad = True
                    
                    # Instantiate GradCAM generator
                    cam_generator = GradCAM(py_model, target_layer)
                    heatmap = cam_generator(img_tensor, target_class=0)  # 0 is 'cancer'
                    cam_generator.release()
                    
                    if heatmap is not None:
                        # Convert PIL to BGR OpenCV image
                        img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                        h, w, _ = img_cv.shape
                        
                        # Resize heatmap
                        heatmap_resized = cv2.resize(heatmap, (w, h))
                        
                        # Create jet colormap
                        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                        
                        # Superimpose original and heatmap
                        superimposed = cv2.addWeighted(img_cv, 0.65, heatmap_color, 0.35, 0)
                        
                        # Save image
                        heatmap_filename = f"gradcam_{safe_email}_{timestamp_sec}.png"
                        heatmap_filepath = os.path.join(UPLOAD_FOLDER, heatmap_filename)
                        cv2.imwrite(heatmap_filepath, superimposed)
                        heatmap_url = f"/static/uploads/{heatmap_filename}"
                        print(f"[INFO] Grad-CAM heatmap generated: {heatmap_url}")
        except Exception as cam_err:
            print(f"[WARN] Failed to generate Grad-CAM: {cam_err}")
            traceback.print_exc()

        # Construct and log case history
        case_id = f"SCAN-{timestamp_sec % 100000:05d}"
        formatted_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        history_record = {
            "case_id": case_id,
            "timestamp": formatted_time,
            "patient_name": patient_name,
            "patient_age": patient_age,
            "anatomical_site": anatomical_site,
            "prediction": top1_label,
            "confidence": round(top1_conf * 100, 2),
            "class_probabilities": class_probs,
            "image_url": image_url,
            "heatmap_url": heatmap_url,
            "notes": ""
        }

        email = session.get("email")
        users = load_users()
        if email in users:
            if "history" not in users[email]:
                users[email]["history"] = []
            users[email]["history"].append(history_record)
            save_users(users)

        return jsonify({
            "prediction": top1_label,
            "confidence": round(top1_conf * 100, 2),
            "class_probabilities": class_probs,
            "image_b64": img_b64,
            "image_url": image_url,
            "heatmap_url": heatmap_url,
            "case_id": case_id,
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