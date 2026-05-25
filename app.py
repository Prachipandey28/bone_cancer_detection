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

from dotenv import load_dotenv
load_dotenv()

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


@app.route("/history/delete", methods=["POST"])
def history_delete():
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized session. Please login."}), 401
        
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing payload"}), 400
        
    case_id = data.get("case_id")
    if not case_id:
        return jsonify({"error": "Case ID is required"}), 400
        
    email = session.get("email")
    users = load_users()
    
    if email in users and "history" in users[email]:
        history_list = users[email]["history"]
        updated_history = [record for record in history_list if record["case_id"] != case_id]
        
        if len(updated_history) < len(history_list):
            users[email]["history"] = updated_history
            save_users(users)
            return jsonify({"status": "success", "message": f"Case record {case_id} deleted successfully."})
            
    return jsonify({"error": "Case record not found."}), 404

@app.route("/case/<case_id>/insights")
def case_insights(case_id):
    if not session.get("logged_in"):
        return redirect(url_for("login"))
        
    email = session.get("email")
    users = load_users()
    record = None
    if email in users and "history" in users[email]:
        for rec in users[email]["history"]:
            if rec["case_id"] == case_id:
                record = rec
                break
                
    if not record:
        return "Case record not found.", 404
        
    return render_template("insights.html", record=record)


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

def generate_simulated_insights(prediction, confidence, patient_name, patient_age, anatomical_site):
    is_malignant = prediction.lower() == "cancer"
    
    # English Simulated Data
    if is_malignant:
        en_observations = (
            f"The deep neural diagnostic pipeline detected a localized malignant bone pattern at the {anatomical_site} "
            f"with {confidence:.1f}% confidence. The scan indicates cortical thinning and potentially osteolytic signatures "
            f"characteristic of high-grade bone lesions. Given the patient's demographic ({patient_age}), "
            f"immediate clinical and radiological correlation is indicated."
        )
        en_precautions = [
            "Pathological Fracture Risk: Enforce strict non-weight-bearing or protective support protocols on the affected limb.",
            "Avoid physical manipulation or deep tissue massage of the local area to minimize micro-fracture risks.",
            "Monitor patient closely for systemic symptoms including low-grade fever, unexplained weight loss, or persistent nocturnal bone pain.",
            "Restrict strenuous physical exercises or impact activities until clinical staging is finalized."
        ]
        en_recommendations = [
            f"Order an urgent High-Resolution Contrast-Enhanced MRI or CT scan of the {anatomical_site} for detailed soft-tissue staging.",
            "Secure a referral to a certified Orthopedic Oncologist for multidisciplinary tumor board (MDT) staging.",
            "Schedule a clinical core needle biopsy of the identified lesion under image-guided navigation.",
            "Conduct comprehensive blood workups, including Serum Alkaline Phosphatase (ALP) and Lactate Dehydrogenase (LDH)."
        ]
    else:
        en_observations = (
            f"Neural telemetry shows uniform cortical density and smooth joint boundaries at the {anatomical_site} "
            f"with {confidence:.1f}% confidence. No active osteolytic bone destruction or classic malignant osteosarcoma "
            f"signatures are identified. Cortical borders appear intact with normal skeletal alignment."
        )
        en_precautions = [
            "Maintain standard musculoskeletal support; avoid unnecessary strain if pain persists.",
            "Advise the patient to monitor for any developing swelling, redness, or localized heat over the joint.",
            "Follow general joint protection guidelines and avoid sudden high-impact loading if recovering from soft-tissue trauma.",
            "Report immediately if there is any sudden increase in localized pain, especially at rest or during the night."
        ]
        en_recommendations = [
            "Recommend clinical evaluation by an orthopedic specialist or physical therapist if mechanical pain persists.",
            "Utilize conservative management protocols (e.g., R.I.C.E.) for suspected joint sprain or ligamentous strain.",
            "Schedule a standard follow-up radiograph in 4-6 weeks if symptoms fail to resolve with conservative therapy.",
            "Consider soft-tissue imaging (Ultrasound or MRI) if there is clinical suspicion of a ligament, tendon, or meniscus tear."
        ]
        
    # Hindi Simulated Data
    if is_malignant:
        hi_observations = (
            f"डीप न्यूरल डायग्नोस्टिक पाइपलाइन ने {anatomical_site} पर {confidence:.1f}% विश्वास के साथ एक स्थानीय घातक हड्डी के स्वरूप (कैंसर के लक्षण) का पता लगाया है। स्कैन कॉर्टिकल क्षति और ऑस्टियोलाइटिक विशेषताओं को दर्शाता है जो उच्च-श्रेणी के हड्डी के घावों की विशेषता हैं। रोगी के जनसांख्यिकीय ({patient_age}) को देखते हुए, तत्काल नैदानिक और रेडियोलॉजिकल संबंध स्थापित करना संकेतित है।"
        )
        hi_precautions = [
            "हड्डी टूटने का खतरा (Pathological Fracture Risk): प्रभावित अंग पर वजन न डालें और चलने-फिरने के लिए आवश्यक सुरक्षा उपकरण का उपयोग करें।",
            "प्रभावित क्षेत्र की मालिश (Deep Tissue Massage) या किसी भी प्रकार के शारीरिक खिंचाव से बचें, ताकि सूक्ष्म-फ्रैक्चर के जोखिम को कम किया जा सके।",
            "मरीज की कड़ी निगरानी करें: बुखार, वजन में अचानक कमी, या रात में होने वाले तेज हड्डी के दर्द जैसे लक्षणों पर ध्यान दें।",
            "नैदानिक जाँच पूरी होने तक किसी भी प्रकार की भारी शारीरिक कसरत या उच्च-प्रभाव वाली गतिविधियों को पूरी तरह से प्रतिबंधित करें।"
        ]
        hi_recommendations = [
            f"विस्तृत नरम ऊतक जाँच के लिए {anatomical_site} का तत्काल हाई-रिज़ॉल्यूशन कंट्रास्ट-एन्हांस्ड एमआरआई (MRI) या सीटी स्कैन करवाएं।",
            "बहु-विषयक ट्यूमर बोर्ड (MDT) द्वारा जाँच के लिए एक प्रमाणित ऑर्थोपेडिक ऑन्कोलॉजिस्ट (हड्डी कैंसर विशेषज्ञ) के पास तुरंत रेफर करें।",
            "इमेज-निर्देशित नेविगेशन के तहत प्रभावित घाव की क्लीनिकल कोर नीडल बायोप्सी (Biopsy) निर्धारित करें।",
            "सीरम एल्कलाइन फॉस्फेट (ALP) और लैक्टेट डिहाइड्रोजनेज (LDH) सहित व्यापक रक्त जाँच करवाएं।"
        ]
    else:
        hi_observations = (
            f"न्यूरल टेलीमेट्री {anatomical_site} पर {confidence:.1f}% विश्वास के साथ समान कॉर्टिकल घनत्व और सुचारू जोड़ सीमाओं को दर्शाती है। कोई सक्रिय ऑस्टियोलाइटिक हड्डी विनाश या क्लासिक ऑस्टियोसारकोमा संकेत नहीं पाए गए हैं। स्केलेटल संरेखण सामान्य है।"
        )
        hi_precautions = [
            "सामान्य मस्कुलोस्केलेटल सहायता बनाए रखें; दर्द बने रहने पर अनावश्यक खिंचाव से बचें।",
            "रोगी को सलाह दें कि वे प्रभावित जोड़ पर सूजन, लाली या गर्मी विकसित होने की निगरानी करें।",
            "जोड़ सुरक्षा दिशानिर्देशों का पालन करें और अचानक से बहुत भारी वजन उठाने से बचें।",
            "यदि रात में या आराम करते समय दर्द में अचानक वृद्धि होती है, तो तुरंत रिपोर्ट करें।"
        ]
        hi_recommendations = [
            "यांत्रिक दर्द बने रहने पर ऑर्थोपेडिक विशेषज्ञ या फिजियोथेरेपिस्ट द्वारा मूल्यांकन की सिफारिश करें।",
            "जोड़ मोच या लिगामेंट खिंचाव के लिए सामान्य रूढ़िवादी प्रबंधन (R.I.C.E.) प्रोटोकॉल का उपयोग करें।",
            "यदि रूढ़िवादी चिकित्सा से लक्षण ठीक नहीं होते हैं, तो 4-6 सप्ताह में एक सामान्य अनुवर्ती एक्स-रे निर्धारित करें।",
            "लिगामेंट, कण्डरा या मेनिस्कस फटने की क्लीनिकल आशंका होने पर नरम-ऊतक इमेजिंग (अल्ट्रासाउंड या एमआरआई) पर विचार करें।"
        ]
        
    return {
        "en": {
            "observations": en_observations,
            "precautions": en_precautions,
            "recommendations": en_recommendations
        },
        "hi": {
            "observations": hi_observations,
            "precautions": hi_precautions,
            "recommendations": hi_recommendations
        },
        "simulated": True
    }


@app.route("/gemini_analysis", methods=["POST"])
def gemini_analysis():
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized session. Please login."}), 401
        
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing payload"}), 400
        
    case_id = data.get("case_id")
    prediction = data.get("prediction")
    confidence = data.get("confidence")
    class_probabilities = data.get("class_probabilities", {})
    patient_name = data.get("patient_name", "PAT-TEMP")
    patient_age = data.get("patient_age", "—")
    anatomical_site = data.get("anatomical_site", "Other / General")
    
    if not case_id or not prediction:
        return jsonify({"error": "Missing required fields"}), 400
        
    # Get API key from environment
    api_key = os.environ.get("GEMINI_API_KEY")  
    
    insights = None
    is_simulated = False
    
    if not api_key:
        print("[INFO] GEMINI_API_KEY env variable not set. Generating simulated clinical insights.")
        insights = generate_simulated_insights(prediction, confidence, patient_name, patient_age, anatomical_site)
        is_simulated = True
    else:
        try:
            import urllib.request
            
            prompt = f"""
You are an expert orthopedic oncologist and senior radiologist clinical assistant.
Analyze the following deep learning computer vision scan telemetry and patient intake metadata:

[SCAN TELEMETRY]
- Neural Classification Verdict: {prediction.upper()}
- Model Staging Confidence: {confidence:.2f}%
- Neural Probabilities: {json.dumps(class_probabilities)}

[PATIENT METADATA]
- Patient Name/Ref: {patient_name}
- Patient Age/Gender: {patient_age}
- Anatomical Joint/Site: {anatomical_site}

Provide a JSON object containing both English ('en') and Hindi ('hi') clinical insights.
For 'en' (English) and 'hi' (Hindi), generate:
1. 'observations': A highly professional, context-specific summary of radiological observations (2-3 sentences). Discuss potential implications of the neural findings for this specific joint/anatomical site ({anatomical_site}) and patient age ({patient_age}). Mention any standard features or anomalies associated with {prediction} scans. Write the 'hi' version in clear, professional medical Hindi.
2. 'precautions': A list of 3-4 highly relevant precautions. For MALIGNANT, include strict instructions like non-weight-bearing protocols if pathological fracture risk exists, avoiding vigorous manipulation, and immediate oncological isolation. For NORMAL, include general safety, monitoring for developing symptoms, and proper ergonomics. Write the 'hi' version in clear, professional medical Hindi.
3. 'recommendations': A list of 3-4 advanced diagnostic next-steps or clinical routings. For MALIGNANT, recommend contrast MRI, core needle biopsy, orthopedic oncology referral, blood workups (ALP, LDH). For NORMAL, recommend clinical follow-up if symptoms persist, physical therapy if mechanical, or routine surveillance. Write the 'hi' version in clear, professional medical Hindi.

Maintain a strict, objective, professional medical tone. Do NOT include generic disclaimers in the generated fields, as a global system disclaimer is already rendered on the report.
"""
            
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
            headers = {
                "Content-Type": "application/json"
            }
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseSchema": {
                        "type": "OBJECT",
                        "properties": {
                            "en": {
                                "type": "OBJECT",
                                "properties": {
                                    "observations": {
                                        "type": "STRING",
                                        "description": "Professional clinical observations in English."
                                    },
                                    "precautions": {
                                        "type": "ARRAY",
                                        "items": { "type": "STRING" },
                                        "description": "List of key precautions in English."
                                    },
                                    "recommendations": {
                                        "type": "ARRAY",
                                        "items": { "type": "STRING" },
                                        "description": "List of clinical recommendations in English."
                                    }
                                },
                                "required": ["observations", "precautions", "recommendations"]
                            },
                            "hi": {
                                "type": "OBJECT",
                                "properties": {
                                    "observations": {
                                        "type": "STRING",
                                        "description": "Professional clinical observations translated or written directly in clear, formal Hindi."
                                    },
                                    "precautions": {
                                        "type": "ARRAY",
                                        "items": { "type": "STRING" },
                                        "description": "List of key precautions in formal Hindi."
                                    },
                                    "recommendations": {
                                        "type": "ARRAY",
                                        "items": { "type": "STRING" },
                                        "description": "List of clinical recommendations in formal Hindi."
                                    }
                                },
                                "required": ["observations", "precautions", "recommendations"]
                            }
                        },
                        "required": ["en", "hi"]
                    }
                }
            }
            
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=req_data, headers=headers, method="POST")
            
            with urllib.request.urlopen(req, timeout=15) as response:
                res_data = response.read().decode("utf-8")
                res_json = json.loads(res_data)
                
                # Extract structured text response
                text_content = res_json["candidates"][0]["content"]["parts"][0]["text"]
                insights = json.loads(text_content)
                insights["simulated"] = False
                
        except Exception as e:
            print(f"[WARN] Gemini API call failed: {e}. Falling back to simulated insights.")
            traceback.print_exc()
            insights = generate_simulated_insights(prediction, confidence, patient_name, patient_age, anatomical_site)
            is_simulated = True
            
    # Save the insights to the user's history in users.json
    email = session.get("email")
    users = load_users()
    if email in users and "history" in users[email]:
        for record in users[email]["history"]:
            if record["case_id"] == case_id:
                record["gemini_analysis"] = insights
                save_users(users)
                break
                
    return jsonify({
        "status": "success",
        "insights": insights,
        "is_simulated": is_simulated
    })


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
    })


def generate_simulated_chat_response(query, current_scan, current_insights, language):
    query_lower = query.lower()
    
    # Context Staging
    is_malignant = False
    if current_scan and current_scan.get('prediction', '').lower() == 'cancer':
        is_malignant = True
        
    patient_ref = current_scan.get('patient_name', 'PAT-TEMP') if current_scan else 'PAT-TEMP'
    anatomical = current_scan.get('anatomical_site', 'affected limb') if current_scan else 'affected area'
    
    is_hi = language == 'hi' or any(c in query for c in ['कैंसर', 'इलाज', 'दवा', 'क्या', 'हड्डी', 'जांच', 'सावधानी', 'नमस्ते', 'बचाव', 'दर्द', 'बीमारी'])

    if is_hi:
        if any(w in query_lower for w in ["नमस्ते", "हेलो", "प्रणाम", "hey", "hello", "hi"]):
            return f"नमस्ते! मैं आपका ऑस्टियोस्कैन एआई क्लीनिकल को-पायलट हूँ। {patient_ref} के संदर्भ में, मैं आपकी सहायता के लिए तैयार हूँ। आप मुझसे इस केस की जांच, स्टेजिंग, या आवश्यक सावधानियों के बारे में पूछ सकते हैं।"
        elif any(w in query_lower for w in ["कैंसर", "घातक", "बीमारी", "कैसा", "क्या है", "दिक्कत", "समस्या", "रोग", "malignant", "issue", "problem"]):
            if is_malignant:
                return f"ऑस्टियोस्कैन न्यूरल कोर ने {patient_ref} के {anatomical} पर एक घातक ट्यूमर पैटर्न (Malignant bone cancer pattern) की पहचान की है। स्कैन कॉर्टिकल थिनिंग और ऑस्टियोलाइटिक सीमाओं को दर्शाता है, जो तत्काल रेफरल और बायोप्सी की मांग करता है।"
            else:
                return f"रोगी {patient_ref} का स्कैन सामान्य दिखाई दे रहा है। {anatomical} पर हड्डी की घनत्व और जोड़ की सीमाएं सुचारू हैं, और कैंसर/ऑस्टियोसारकोमा के कोई सक्रिय संकेत नहीं हैं।"
        elif any(w in query_lower for w in ["सावधानी", "बचाव", "परहेज", "रोक", "precaution"]):
            if is_malignant:
                return "घातक पैटर्न के लिए अत्यंत महत्वपूर्ण नैदानिक सावधानियां:\n1. पैथोलॉजिकल फ्रैक्चर के जोखिम के कारण प्रभावित अंग पर भारी वजन बिल्कुल न डालें।\n2. प्रभावित क्षेत्र की मालिश, भारी खिंचाव या अनावश्यक शारीरिक हेरफेर से बचें।\n3. रात के दर्द या लगातार बुखार जैसे लक्षणों की कड़ी निगरानी करें।"
            else:
                return "सामान्य स्कैन के लिए सावधानियां:\n1. अचानक भारी भार उठाने या अनावश्यक मस्कुलोस्केलेटल तनाव से बचें।\n2. प्रभावित जोड़ पर असामान्य गर्मी या सूजन की निगरानी करें।"
        elif any(w in query_lower for w in ["बायोप्सी", "biopsy", "mri", "जांच", "एक्सरे", "xray", "टेस्ट", "scan"]):
            return f"निदान की अंतिम पुष्टि के लिए इमेज-निर्देशित क्लीनिकल कोर नीडल बायोप्सी (Biopsy) स्वर्ण मानक है। आस-पास के कोमल ऊतकों में फैलाव की जांच के लिए {anatomical} का तत्काल कंट्रास्ट-एन्हांस्ड एमआरआई (MRI) कराना आवश्यक है।"
        elif any(w in query_lower for w in ["इलाज", "ट्रीटमेंट", "treatment", "डॉक्टर", "ऑन्कोलॉजिस्ट"]):
            return f"संदीग्ध घातक निष्कर्षों को देखते हुए, रोगी {patient_ref} को तुरंत एक प्रमाणित ऑर्थोपेडिक ऑन्कोलॉजिस्ट और मल्टी-डिसिप्लिनरी ट्यूमर बोर्ड (MDT) के पास रेफर किया जाना चाहिए। साथ ही रक्त जांच (Serum ALP, LDH) भी कराई जानी चाहिए।"
        else:
            status_text = "घातक पैटर्न" if is_malignant else "सामान्य हड्डी संरचना"
            return f"नमस्ते! {patient_ref} के संदर्भ में ({status_text} - {anatomical}): मैं आपकी किस प्रकार सहायता कर सकता हूँ? आप मुझसे Staging (बायोप्सी/एमआरआई) या आवश्यक Precautions (सावधानियां) के बारे में पूछ सकते हैं।"
    else:
        # English replies
        if any(w in query_lower for w in ["hello", "hi", "hey", "greetings"]):
            return f"Hello! I am your OsteoScan AI Co-Pilot. I am statefully aware of patient {patient_ref}'s active telemetry. How can I assist you with clinical correlations, guidelines, or diagnostic next steps today?"
        elif any(w in query_lower for w in ["cancer", "malignant", "issue", "problem", "disease", "diagnosis", "verdict", "what", "find", "happen", "report", "image", "scan", "xray", "x-ray"]):
            if is_malignant:
                return f"The active diagnostics for {patient_ref} show a MALIGNANT bone pattern at the {anatomical} with {current_scan.get('confidence', 59.8):.1f}% confidence. Key features indicate active osteolytic bone destruction and cortical thinning, requiring urgent isolation and clinical staging."
            else:
                return f"The active skeletal borders for {patient_ref} appear intact. Smooth joint boundaries and uniform cortical density at the {anatomical} indicate a uniform normal skeletal structure with no signs of active osteosarcoma."
        elif any(w in query_lower for w in ["precaution", "fracture", "safety", "careful", "avoid", "protect"]):
            if is_malignant:
                return "Critical clinical precautions for active malignant findings:\n1. Strict non-weight-bearing support or protective immobilization on the affected limb to prevent pathological fracture.\n2. Strictly avoid heavy physical manipulation, deep tissue mobilization, or heat therapies.\n3. Closely monitor for oncology red-flags (constant rest pain, spikes in fever)."
            else:
                return "Standard clinical precautions for normal bone radiographs:\n1. Maintain progressive recovery; avoid heavy impact loading if mechanical pain persists.\n2. Advise the patient to monitor for local warmth or erythema."
        elif any(w in query_lower for w in ["biopsy", "mri", "ct", "ultrasound", "imaging", "test", "next step", "guideline"]):
            return f"For suspected high-grade bone lesions at the {anatomical}, the standard clinical pathway is a High-Resolution Contrast-Enhanced MRI to stage local margins, followed by an image-guided core needle biopsy for pathological confirmation."
        elif any(w in query_lower for w in ["treatment", "care", "therapy", "doctor", "oncologist", "referral", "cure", "do"]):
            return f"For suspicious oncological scans, establish immediate referral to a certified Orthopedic Oncologist and coordinate a multidisciplinary tumor board (MDT). Staging baseline blood markers (ALP, LDH) should also be checked."
        else:
            status_text = "malignant bone pattern" if is_malignant else "uniform normal bone structure"
            return f"Regarding patient {patient_ref} ({status_text} observed at the {anatomical}): I can provide details on Staging Guidelines (Biopsy/MRI), critical Precautions, or Oncology Referral pathways. What specific details would you like to review?"


@app.route("/chat", methods=["POST"])
def chat():
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized session. Please login."}), 401
        
    data = request.get_json()
    if not data or not data.get("message"):
        return jsonify({"error": "Missing message content."}), 400
        
    user_message = data.get("message")
    chat_history = data.get("history", [])
    current_scan = data.get("current_scan")
    current_insights = data.get("current_insights")
    active_language = data.get("language", "en")
    
    api_key = os.environ.get("GEMINI_API_KEY")
    
    # System Context
    system_context = """
You are "OsteoScan AI Co-Pilot", a senior orthopedic oncology clinical assistant chatbot designed by the Google DeepMind team.
Your goal is to support the radiologist or clinician in analyzing bone scans, answering their clinical, diagnostic, and pathological questions, and providing medical suggestions.

Maintain a professional, helpful, objective, and empathetic clinical medical tone. 
Always make it clear that your answers are AI decision-support insights and the final diagnosis lies with a licensed medical practitioner.
"""

    if current_scan:
        system_context += f"""
Current Active Case Context:
- Patient Name/Ref: {current_scan.get('patient_name', 'PAT-TEMP')}
- Age/Gender: {current_scan.get('patient_age', '—')}
- Anatomical Joint/Site: {current_scan.get('anatomical_site', 'General')}
- YOLOv8 Classification Prediction: {current_scan.get('prediction', '—').upper()}
- Staging Model Confidence: {current_scan.get('confidence', 0):.2f}%
"""
    if current_insights:
        system_context += f"""
Gemini Clinical Assistant Insights:
- Observations: {current_insights.get('observations', '—')}
- Critical Precautions: {", ".join(current_insights.get('precautions', []))}
- Clinical Recommendations: {", ".join(current_insights.get('recommendations', []))}
"""

    system_context += f"""
Respond in the active selected language if possible ({'Hindi' if active_language == 'hi' else 'English'}), or match the language used by the clinician in their query. Keep your explanations clear, structured, and clinically precise.
"""

    if api_key:
        try:
            import requests

            contents = []

            # Append history in Gemini API contents format
            for turn in chat_history:
                role = "user" if turn["role"] == "user" else "model"
                contents.append({
                    "role": role,
                    "parts": [{"text": turn["text"]}]
                })

            contents.append({
                "role": "user",
                "parts": [{"text": user_message}]
            })

            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"

            payload = {
                "contents": contents,
                "systemInstruction": {
                    "parts": [{"text": system_context}]
                },
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 800
                }
            }

            response = requests.post(url, json=payload, timeout=15)

            print(f"[DEBUG] Gemini status: {response.status_code}")
            print(f"[DEBUG] Gemini response preview: {response.text[:300]}")

            if response.status_code != 200:
                raise Exception(f"Gemini returned {response.status_code}: {response.text}")

            res_json = response.json()
            ai_text = res_json["candidates"][0]["content"]["parts"][0]["text"]

            return jsonify({
                "status": "success",
                "reply": ai_text,
                "is_simulated": False
            })

        except Exception as e:
            print(f"[ERROR] Chat Gemini API failed: {e}")
            traceback.print_exc()

    # Fallback to simulated response
    reply = generate_simulated_chat_response(user_message, current_scan, current_insights, active_language)
    return jsonify({
        "status": "success",
        "reply": reply,
        "is_simulated": True
    })

# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)