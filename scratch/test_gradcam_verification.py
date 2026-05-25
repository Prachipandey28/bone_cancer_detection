import urllib.request
import json
import os

print("[VERIFICATION] Running Grad-CAM heatmap specific verification...")
BASE_URL = "http://127.0.0.1:8080"

# 1. Login
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor())
urllib.request.install_opener(opener)
login_payload = {
    "username": "prachipandey1528@gmail.com",
    "password": "prachi15"
}
login_req = urllib.request.Request(
    f"{BASE_URL}/login",
    data=json.dumps(login_payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

try:
    with urllib.request.urlopen(login_req) as resp:
        login_res = json.loads(resp.read().decode("utf-8"))
        assert login_res.get("status") == "success"
except Exception as e:
    print("[ERROR] Login failed:", e)
    raise

# 2. Predict X-ray Scan
test_image_path = "/Users/avinashpandey/bone_cancer_detection/static/uploads/scan_prachipandey1528_gmail_com_1779645703.jpeg"
with open(test_image_path, "rb") as f:
    file_bytes = f.read()

boundary = "----WebKitFormBoundaryClinicalScanner2026"
headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}

body_parts = [
    f"--{boundary}\r\nContent-Disposition: form-data; name=\"patientName\"\r\n\r\nVERIFY-TEST-PATIENT\r\n".encode("utf-8"),
    f"--{boundary}\r\nContent-Disposition: form-data; name=\"patientAge\"\r\n\r\n45 M\r\n".encode("utf-8"),
    f"--{boundary}\r\nContent-Disposition: form-data; name=\"anatomicalSite\"\r\n\r\nDistal Femur\r\n".encode("utf-8"),
    f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"test_xray.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n".encode("utf-8"),
    file_bytes,
    f"\r\n--{boundary}--\r\n".encode("utf-8")
]
body = b"".join(body_parts)

predict_req = urllib.request.Request(
    f"{BASE_URL}/predict",
    data=body,
    headers=headers
)

try:
    with urllib.request.urlopen(predict_req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        print("[VERIFICATION] /predict Response:")
        print(json.dumps(data, indent=2))
        
        # Assertions
        assert data.get("status") == "success", "Status not success!"
        assert "heatmap_url" in data, "heatmap_url not in response!"
        assert data["heatmap_url"] is not None, "heatmap_url is null!"
        assert data["heatmap_url"].startswith("/static/uploads/gradcam_"), "heatmap_url is not named correctly!"
        
        # Verify file existence on disk
        relative_path = data["heatmap_url"].lstrip("/")
        full_filepath = os.path.join("/Users/avinashpandey/bone_cancer_detection", relative_path)
        assert os.path.exists(full_filepath), f"Heatmap file does not exist at {full_filepath}!"
        assert os.path.getsize(full_filepath) > 0, f"Heatmap file size is 0 bytes!"
        print(f"[VERIFICATION] Heatmap file validated successfully! Size: {os.path.getsize(full_filepath)} bytes.")
        
except Exception as e:
    print("[ERROR] Prediction verification failed:", e)
    raise

print("\n[VERIFICATION SUCCESS] GRAD-CAM HEATMAP VERIFIED AND FULLY OPERATIONAL! 🎉")
