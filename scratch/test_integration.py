import urllib.request
import json
import os

print("[TEST] Launching quick standard integration test (ZERO DEPENDENCIES)...")
BASE_URL = "http://127.0.0.1:8080"

# Create a cookie handler to persist session cookies across requests (acts like requests.Session)
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor())
urllib.request.install_opener(opener)

# 1. Login as the newly created doctor account
login_payload = {
    "username": "prachipandey1528@gmail.com",
    "password": "prachi15"
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

# 2. Perform predictive scan with custom patient details (Multipart Form-Data manually crafted)
test_image_path = "/Users/avinashpandey/bone_cancer_detection/static/uploads/scan_prachipandey1528_gmail_com_1779645703.jpeg"
print(f"[TEST] Staging upload image: {test_image_path}")

if not os.path.exists(test_image_path):
    raise FileNotFoundError(f"Test image not found at {test_image_path}")

with open(test_image_path, "rb") as f:
    file_bytes = f.read()

# Build multipart form data body manually to avoid using requests
boundary = "----WebKitFormBoundaryClinicalScanner2026"
headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}

body_parts = [
    f"--{boundary}\r\nContent-Disposition: form-data; name=\"patientName\"\r\n\r\nPAT-9988-X\r\n".encode("utf-8"),
    f"--{boundary}\r\nContent-Disposition: form-data; name=\"patientAge\"\r\n\r\n52 F\r\n".encode("utf-8"),
    f"--{boundary}\r\nContent-Disposition: form-data; name=\"anatomicalSite\"\r\n\r\nPelvic Ring\r\n".encode("utf-8"),
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
        assert "case_id" in predict_data, "Case ID missing from response!"
except Exception as e:
    print(f"[ERROR] Prediction request failed: {e}")
    raise

case_id = predict_data["case_id"]
print(f"[TEST] Staging observations commit for Case ID: {case_id}")

# 3. Commit Radiologist Notes
notes_payload = {
    "case_id": case_id,
    "notes": "Pelvic cortical boundaries are intact. Normal joint density profiles mapped. Clear of tumor features."
}
notes_req = urllib.request.Request(
    f"{BASE_URL}/history/notes",
    data=json.dumps(notes_payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

try:
    with urllib.request.urlopen(notes_req) as resp:
        notes_res = json.loads(resp.read().decode("utf-8"))
        print(f"[TEST] Notes Status: {resp.status}, Response: {notes_res}")
        assert notes_res.get("status") == "success", "Notes signing failed!"
except Exception as e:
    print(f"[ERROR] Notes signing failed: {e}")
    raise

# 4. Fetch and assert Case History
history_req = urllib.request.Request(f"{BASE_URL}/history")

try:
    with urllib.request.urlopen(history_req) as resp:
        history_data = json.loads(resp.read().decode("utf-8"))
        print(f"[TEST] History Status: {resp.status}")
        print(f"[TEST] Retrieved Latest History Item: {history_data[0]}")
        assert len(history_data) > 0, "Scan history array empty!"
        assert history_data[0]["case_id"] == case_id, "Latest Case ID mismatch in database!"
        assert history_data[0]["patient_name"] == "PAT-9988-X", "Patient name mismatch!"
        assert history_data[0]["notes"] == notes_payload["notes"], "Case notes mismatch in database!"
except Exception as e:
    print(f"[ERROR] History assertion failed: {e}")
    raise

print("\n[SUCCESS] E2E PERSISTENT SCAN HISTORY LOG & TELEMETRY FLOW INTEGRATION PASSED 100%! 🎉")
