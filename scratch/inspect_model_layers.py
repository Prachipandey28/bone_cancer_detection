import os
from ultralytics import YOLO

MODEL_PATH = "weights/best.pt"
model = YOLO(MODEL_PATH, task="classify")
py_model = model.model

print("Layer 8:", py_model.model[8])
print("Layer 9:", py_model.model[9])

# Let's inspect Layer 9 attributes
l9 = py_model.model[9]
print("\n--- Layer 9 Classify Info ---")
for k, v in l9.__dict__.items():
    if not k.startswith('_'):
        print(f"Attribute {k}: {v}")
