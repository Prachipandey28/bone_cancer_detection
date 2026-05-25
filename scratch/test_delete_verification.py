import urllib.request
import json
import os

print("[VERIFICATION] Running `/history/delete` specific verification...")
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

# 2. Predict X-ray Scan to create a dummy record
test_image_path = "/Users/avinashpandey/bone_cancer_detection/static/uploads/scan_prachipandey1528_gmail_com_1779645703.jpeg"
with open(test_image_path, "rb") as f:
    file_bytes = f.read()

boundary = "----WebKitFormBoundaryClinicalScanner2026"
headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}

body_parts = [
    f"--{boundary}\r\nContent-Disposition: form-data; name=\"patientName\"\r\n\r\nTO-DELETE-PATIENT\r\n".encode("utf-8"),
    f"--{boundary}\r\nContent-Disposition: form-data; name=\"patientAge\"\r\n\r\n45 M\r\n".encode("utf-8"),
    f"--{boundary}\r\nContent-Disposition: form-data; name=\"anatomicalSite\"\r\n\r\nPelvic Ring\r\n".encode("utf-8"),
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
        case_id = data.get("case_id")
        print(f"[VERIFICATION] Staged new dummy case: {case_id}")
        assert data.get("status") == "success"
except Exception as e:
    print("[ERROR] Prediction failed:", e)
    raise

# 3. Verify case is present in History
history_req = urllib.request.Request(f"{BASE_URL}/history")
try:
    with urllib.request.urlopen(history_req) as resp:
        history_data = json.loads(resp.read().decode("utf-8"))
        case_ids = [item["case_id"] for item in history_data]
        assert case_id in case_ids, f"Case {case_id} not found in history!"
        print(f"[VERIFICATION] Confirmed case {case_id} is present in History list.")
except Exception as e:
    print("[ERROR] History verification failed:", e)
    raise

# 4. Trigger Delete Route
delete_payload = { "case_id": case_id }
delete_req = urllib.request.Request(
    f"{BASE_URL}/history/delete",
    data=json.dumps(delete_payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)
try:
    with urllib.request.urlopen(delete_req) as resp:
        delete_res = json.loads(resp.read().decode("utf-8"))
        print("[VERIFICATION] Delete response:", delete_res)
        assert delete_res.get("status") == "success"
except Exception as e:
    print("[ERROR] Delete request failed:", e)
    raise

# 5. Verify case is no longer present in History
try:
    with urllib.request.urlopen(history_req) as resp:
        history_data = json.loads(resp.read().decode("utf-8"))
        case_ids = [item["case_id"] for item in history_data]
        assert case_id not in case_ids, f"Case {case_id} still found in history after deletion!"
        print(f"[VERIFICATION] Confirmed case {case_id} is successfully REMOVED from History list.")
except Exception as e:
    print("[ERROR] History deletion verification failed:", e)
    raise

print("\n[VERIFICATION SUCCESS] CASE DELETION SECURE PIPELINE VERIFIED AND FULLY OPERATIONAL! 🎉")
