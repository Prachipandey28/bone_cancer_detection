# OsteoScan — Bone Cancer Detection Web App

YOLOv8-based X-ray classifier (cancer vs normal) with a Flask backend and a standalone HTML/CSS/JS frontend.

---

## Project Structure

```
bone_cancer_app/
├── app.py                  ← Flask server
├── templates/
│   └── index.html          ← Frontend UI
├── weights/
│   └── best.pt             ← ⚠ Put your trained YOLOv8 weights here
└── static/
    └── uploads/            ← Auto-created at runtime
```

---

## Setup

### 1. Install dependencies

```bash
pip install flask ultralytics pillow
```

### 2. Add your model weights

Copy the `best.pt` file from your Colab training run into the `weights/` folder:

```
weights/best.pt
```

The model path can also be overridden via environment variable:

```bash
export MODEL_PATH=/path/to/custom/best.pt
```

### 3. Run the server

```bash
python app.py
```

Then open **http://localhost:5000** in your browser.

---

## API Endpoints

| Method | Path       | Description                                    |
|--------|------------|------------------------------------------------|
| GET    | `/`        | Serve the frontend HTML                        |
| POST   | `/predict` | Upload an image; returns JSON prediction       |
| GET    | `/health`  | Check if model is loaded                       |

### `/predict` — request

```
POST /predict
Content-Type: multipart/form-data

file=<image file>
```

### `/predict` — response

```json
{
  "status": "success",
  "prediction": "cancer",
  "confidence": 94.37,
  "class_probabilities": {
    "cancer": 0.9437,
    "normal": 0.0563
  },
  "image_b64": "<base64-encoded PNG thumbnail>"
}
```

---

## Notes

- **Supported image formats:** JPG, JPEG, PNG, BMP, WEBP
- **Max upload size:** 16 MB
- The server runs on port `5000` by default (debug mode on). Disable debug for production.
- The disclaimer in the UI is mandatory — this tool is for **research only**, not clinical diagnosis.