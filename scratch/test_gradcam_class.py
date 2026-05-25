import os
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

# Thread-safe Grad-CAM Class
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        # register hook using register_full_backward_hook to avoid deprecation warnings
        self.forward_handle = self.target_layer.register_forward_hook(self.save_activation)
        self.backward_handle = self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output.detach()
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        
    def __call__(self, x, target_class=0):
        self.model.zero_grad()
        outputs = self.model(x)
        
        # Handle YOLO classification output structure
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

# Load model and run test
MODEL_PATH = "weights/best.pt"
model = YOLO(MODEL_PATH, task="classify")
py_model = model.model
py_model.eval()

target_layer = py_model.model[9].conv

test_image_path = "static/uploads/scan_prachipandey1528_gmail_com_1779645703.jpeg"
if os.path.exists(test_image_path):
    pil_img = Image.open(test_image_path).convert("RGB")
    
    # Preprocess
    from ultralytics.data.augment import classify_transforms
    transform = classify_transforms(224)
    img_tensor = transform(pil_img).unsqueeze(0)
    img_tensor.requires_grad = True
    
    # Generate heatmap
    cam = GradCAM(py_model, target_layer)
    heatmap = cam(img_tensor, target_class=0)
    cam.release()
    
    if heatmap is not None:
        print("Heatmap shape:", heatmap.shape)
        
        # Convert PIL to BGR OpenCV image
        img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        h, w, _ = img_cv.shape
        
        # Resize heatmap
        heatmap_resized = cv2.resize(heatmap, (w, h))
        
        # Colormap
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        
        # Superimpose
        superimposed = cv2.addWeighted(img_cv, 0.65, heatmap_color, 0.35, 0)
        
        output_path = "static/uploads/gradcam_class_test.png"
        cv2.imwrite(output_path, superimposed)
        print("Success! Heatmap saved to:", output_path)
    else:
        print("Heatmap generation failed.")
else:
    print("Test image not found.")
