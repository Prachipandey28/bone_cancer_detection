from ultralytics import YOLO
model = YOLO("weights/best.pt", task="classify")
print("Model names:", model.names)
