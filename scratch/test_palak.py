import urllib.request
import json
import os

print("[TEST] Launching Palak integration test...")
BASE_URL = "http://127.0.0.1:8080"

opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor())
urllib.request.install_opener(opener)

# 1. Login as Palak
login_payload = {
    "username": "palakpandey1528@gmail.com",
    "password": "palak15"
}
print(f"[TEST] Attempting login: {login_payload['username']}")
login_req = urllib.request.Request(
    f"{BASE_URL}/login",
    data=json.dumps(login_payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

try:
    with urllib.request.urlopen(login_req) as resp:
        login_res = json.loads(resp.read().decode("utf-8"))
        print(f"[TEST] Login Status: {resp.status}, Response: {login_res}")
        assert login_res.get("status") == "success", "Login failed!"
except Exception as e:
    print(f"[ERROR] Login request failed: {e}")
    raise

# 2. Upload the exact image that failed
test_image_path = "/Users/avinashpandey/bone_cancer_detection/static/uploads/scan_palakpandey1528_gmail_com_1779646758.jpeg"
print(f"[TEST] Staging upload image: {test_image_path}")

if not os.path.exists(test_image_path):
    raise FileNotFoundError(f"Test image not found at {test_image_path}")

with open(test_image_path, "rb") as f:
    file_bytes = f.read()

boundary = "----WebKitFormBoundaryClinicalScanner2026"
headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}

body_parts = [
    f"--{boundary}\r\nContent-Disposition: form-data; name=\"patientName\"\r\n\r\nPAT-3712-Z\r\n".encode("utf-8"),
    f"--{boundary}\r\nContent-Disposition: form-data; name=\"patientAge\"\r\n\r\n73 M\r\n".encode("utf-8"),
    f"--{boundary}\r\nContent-Disposition: form-data; name=\"anatomicalSite\"\r\n\r\nProximal Femur\r\n".encode("utf-8"),
    f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"test_xray.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n".encode("utf-8"),
    file_bytes,
    f"\r\n--{boundary}--\r\n".encode("utf-8")
]
body = b"".join(body_parts)

print("[TEST] Sending /predict payload...")
predict_req = urllib.request.Request(
    f"{BASE_URL}/predict",
    data=body,
    headers=headers
)

try:
    with urllib.request.urlopen(predict_req) as resp:
        predict_data = json.loads(resp.read().decode("utf-8"))
        print(f"[TEST] Predict Status: {resp.status}")
        print(f"[TEST] Predict Outcome: {predict_data}")
        assert predict_data.get("status") == "success", "Predict route failed!"
except Exception as e:
    print(f"[ERROR] Prediction request failed: {e}")
    raise
