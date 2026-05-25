import urllib.request
import json

try:
    with urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=3) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        print("Server is ONLINE! Response:", data)
except Exception as e:
    print("Server is OFFLINE or not responding:", e)
