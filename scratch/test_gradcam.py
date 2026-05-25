import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO

# 1. Load model and set to eval
MODEL_PATH = "weights/best.pt"
model = YOLO(MODEL_PATH, task="classify")
py_model = model.model
py_model.eval()

# 2. Find target layer for Grad-CAM
# From previous step, py_model.model[9] is the Classify head, and its conv child is the last convolutional layer.
target_layer = py_model.model[9].conv

# Variables to store activations and gradients
activations = None
gradients = None

def forward_hook(module, input, output):
    global activations
    activations = output
    print(f"[Hook] Forward activations shape: {activations.shape}")

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]
    print(f"[Hook] Backward gradients shape: {gradients.shape}")

# Register hooks
forward_handle = target_layer.register_forward_hook(forward_hook)
backward_handle = target_layer.register_backward_hook(backward_hook)

# 3. Load a test image
test_image_path = "static/uploads/scan_prachipandey1528_gmail_com_1779645703.jpeg"
if not os.path.exists(test_image_path):
    # Find any jpeg/png in static/uploads
    import glob
    files = glob.glob("static/uploads/*.*")
    if files:
        test_image_path = files[0]
    else:
        test_image_path = None

print(f"Using image: {test_image_path}")

if test_image_path:
    # Read image using OpenCV and PIL
    pil_img = Image.open(test_image_path).convert("RGB")
    
    # YOLO classification preprocessing: resize to 224x224 (or what the model expects) and normalize
    # Let's see how YOLO classification processes images:
    # Typically, the model expects a tensor of shape [1, 3, 224, 224]
    # Let's inspect the model's preprocessing:
    from ultralytics.data.augment import classify_transforms
    transform = classify_transforms(224)
    img_tensor = transform(pil_img).unsqueeze(0) # [1, 3, 224, 224]
    
    # Enable gradient tracking
    img_tensor.requires_grad = True
    
    # Forward pass through PyTorch model directly (to enable backward gradients)
    logits = py_model(img_tensor)
    print("Logits type:", type(logits))
    if isinstance(logits, tuple):
        print("Tuple len:", len(logits))
        for idx, x in enumerate(logits):
            if torch.is_tensor(x):
                print(f"  Element {idx}: Tensor of shape {x.shape}")
            else:
                print(f"  Element {idx}: {type(x)}")
        # Assuming the first element contains the logits
        logits_tensor = logits[0]
    else:
        logits_tensor = logits
        
    print(f"Logits tensor shape: {logits_tensor.shape}, values: {logits_tensor}")
    
    # The classes in YOLO classification model:
    # 0: 'cancer', 1: 'normal'
    # Let's see the score for class 'cancer' (index 0)
    score = logits_tensor[0, 0]
    
    # Backward pass
    py_model.zero_grad()
    score.backward()
    
    # 4. Compute Grad-CAM
    # Pool the gradients across the spatial dimensions
    # gradients shape: [1, C, H, W]
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3]) # [C]
    
    # Multiply each channel in the activations by its corresponding pooled gradient weight
    # activations shape: [1, C, H, W]
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
        
    # Average the feature maps along the channels dimension to get the heatmap
    heatmap = torch.mean(activations, dim=1).squeeze()
    
    # Apply ReLU to the heatmap to only keep positive features (features that contribute to the class decision)
    heatmap = torch.max(heatmap, torch.zeros_like(heatmap))
    
    # Normalize the heatmap between 0 and 1
    if torch.max(heatmap) > 0:
        heatmap /= torch.max(heatmap)
        
    heatmap = heatmap.detach().cpu().numpy()
    print(f"Heatmap computed successfully! Shape: {heatmap.shape}")
    
    # Save the hooks handles
    forward_handle.remove()
    backward_handle.remove()
    
    # 5. Blend heatmap with original image
    img = cv2.imread(test_image_path)
    h, w, _ = img.shape
    
    # Resize heatmap to match original image size
    heatmap_resized = cv2.resize(heatmap, (w, h))
    
    # Convert heatmap to RGB colors (jet colormap)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    # Superimpose/blend heatmap on original image
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
    
    # Save output
    output_path = "static/uploads/gradcam_test.png"
    cv2.imwrite(output_path, superimposed_img)
    print(f"Saved superimposed image to {output_path}")

else:
    print("No test image found.")
