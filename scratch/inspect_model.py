import os
from ultralytics import YOLO

MODEL_PATH = "weights/best.pt"
print(f"Loading YOLO model from {MODEL_PATH}...")
model = YOLO(MODEL_PATH, task="classify")

print("\n--- YOLO model.model (PyTorch structure) ---")
py_model = model.model
print(type(py_model))

print("\n--- Child Modules ---")
for name, child in py_model.named_children():
    print(f"Child name: {name}, type: {type(child)}")

print("\n--- Sub-modules under model backbone ---")
if hasattr(py_model, "model"):
    for idx, layer in enumerate(py_model.model):
        print(f"Layer {idx}: {type(layer)}")
