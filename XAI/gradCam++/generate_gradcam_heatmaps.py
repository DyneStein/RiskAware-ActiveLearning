import os
import sys
import glob
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn.functional as F
from torchvision import models as tv_models, transforms
from PIL import Image

def seed_everything(seed=42):
    """Ensure research-grade reproducibility across runs and hardware"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed_everything(42)

# Setup directories
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
XAI_DIR = os.path.dirname(CURRENT_DIR)
RESEARCH_DIR = os.path.dirname(XAI_DIR)
DATA_DIR = os.path.join(XAI_DIR, "Data")
MODELS_DIR = os.path.join(XAI_DIR, "Model")
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

class Logger(object):
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

log_path = os.path.join(CURRENT_DIR, "execution_log.txt")
sys.stdout = Logger(log_path)

# Add repo to sys.path for custom models if needed
REPO_DIR = RESEARCH_DIR
if REPO_DIR not in sys.path:
    sys.path.append(REPO_DIR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[{os.path.basename(CURRENT_DIR)}] Using device: {device}")

# Class names mapping
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
DIAGNOSIS_FULL_NAMES = {
    'akiec': 'Actinic Keratosis (AKIEC)',
    'bcc': 'Basal Cell Carcinoma (BCC)',
    'bkl': 'Benign Keratosis (BKL)',
    'df': 'Dermatofibroma (DF)',
    'mel': 'Melanoma (MEL)',
    'nv': 'Melanocytic Nevus (NV)',
    'vasc': 'Vascular Lesion (VASC)'
}

# Image transforms
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load images from Data folder
image_files = []
for ext in ('*.jpg', '*.jpeg', '*.png'):
    image_files.extend(glob.glob(os.path.join(DATA_DIR, ext)))
image_files = sorted(image_files)

# Model files
model_files = sorted(glob.glob(os.path.join(MODELS_DIR, "*.pt")))

print(f"\nDetected number of images: {len(image_files)}")
print(f"Detected number of models: {len(model_files)}\n")

if len(image_files) == 0 or len(model_files) == 0:
    print("Error: Need at least 1 image and 1 model. Exiting.")
    sys.exit(0)

# Model loading helper
try:
    from models.model_factory import create_model as get_model
except Exception:
    def get_model(name, num_classes=7, pretrained=False):
        if 'resnet50' in name:
            m = tv_models.resnet50(pretrained=pretrained)
            m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
            return m
        elif 'densenet169' in name:
            m = tv_models.densenet169(pretrained=pretrained)
            m.classifier = torch.nn.Linear(m.classifier.in_features, num_classes)
            return m
        elif 'efficientnet_b4' in name:
            m = tv_models.efficientnet_b4(pretrained=pretrained)
            m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, num_classes)
            return m
        raise ValueError(f"Unknown model: {name}")

def infer_arch(state_dict):
    """
    Automatically infer the architecture from state_dict keys and tensor shapes.
    First checks layer name patterns, then falls back to feature dimension sizes.
      ResNet-50      → backbone output is 2048
      DenseNet-169   → backbone output is 1664
      EfficientNet-B4→ backbone output is 1792
    """
    keys_str = " ".join(state_dict.keys())

    # --- Pass 1: key name patterns ---
    if "denseblock" in keys_str or "norm5" in keys_str:
        return "densenet169"
    if "layer1" in keys_str or "layer2" in keys_str or "layer3" in keys_str or "layer4" in keys_str:
        return "resnet50"

    # --- Pass 2: feature dimension from head weight shape ---
    for key, tensor in state_dict.items():
        if "head" in key and "weight" in key and tensor.dim() == 2:
            in_features = tensor.shape[1]
            if in_features == 2048:
                return "resnet50"
            elif in_features == 1664:
                return "densenet169"
            elif in_features == 1792:
                return "efficientnet_b4"

    # --- Final fallback ---
    return "efficientnet_b4"

def resolve_target_layer(model, arch_name):
    if 'densenet' in arch_name:
        if hasattr(model, 'backbone') and hasattr(model.backbone, '__getitem__') and hasattr(model.backbone[0], 'norm5'):
            return model.backbone[0].norm5
        elif hasattr(model, 'features') and hasattr(model.features, 'norm5'):
            return model.features.norm5
        for name, mod in model.named_modules():
            if 'norm5' in name:
                return mod
    elif 'resnet' in arch_name:
        if hasattr(model, 'layer4'):
            return model.layer4[-1]
        if hasattr(model, 'backbone') and hasattr(model.backbone, '__getitem__'):
            for i in range(len(model.backbone)-1, -1, -1):
                mod = model.backbone[i]
                if hasattr(mod, '__getitem__') and len(mod) > 0 and not isinstance(mod, (torch.nn.Linear, torch.nn.Conv2d)):
                    try: return mod[-1]
                    except Exception: pass
        for name, mod in model.named_modules():
            if 'layer4' in name and hasattr(mod, '__getitem__') and len(mod) > 0:
                try: return mod[-1]
                except Exception: pass
        last_block = None
        for mod in model.modules():
            if hasattr(mod, 'conv3') or hasattr(mod, 'conv2'):
                last_block = mod
        if last_block is not None:
            return last_block
    elif 'efficientnet' in arch_name:
        if hasattr(model, 'backbone') and hasattr(model.backbone, '__getitem__') and len(model.backbone) > 0:
            feat_seq = model.backbone[0]
            if hasattr(feat_seq, '__getitem__') and len(feat_seq) > 0:
                return feat_seq[-1]
        elif hasattr(model, 'features') and hasattr(model.features, '__getitem__') and len(model.features) > 0:
            return model.features[-1]
        last_conv = None
        for name, mod in model.named_modules():
            if 'features' in name and hasattr(mod, '__getitem__') and len(mod) > 0:
                try: return mod[-1]
                except Exception: pass
            if isinstance(mod, torch.nn.Conv2d):
                last_conv = mod
        if last_conv is not None:
            return last_conv
    raise RuntimeError(f"Could not resolve target layer for architecture: {arch_name}")

class FeatureExtractor:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.fwd_hook = self.target_layer.register_forward_hook(self.save_activation)
        self.bwd_hook = self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.clone() if isinstance(output, torch.Tensor) else output

    def save_gradient(self, module, grad_input, grad_output):
        if grad_output is not None and len(grad_output) > 0 and grad_output[0] is not None:
            self.gradients = grad_output[0].clone()

    def remove_hooks(self):
        self.fwd_hook.remove()
        self.bwd_hook.remove()

def apply_colormap_on_image(org_im, activation_map, colormap_name=cv2.COLORMAP_JET):
    heatmap = cv2.applyColorMap(np.uint8(255 * activation_map), colormap_name)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(org_im) / 255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam), np.uint8(255 * heatmap)

def get_class_logits(out):
    if isinstance(out, (tuple, list)):
        return out[0]
    return out

def compute_gradcam_plus_plus(model, extractor, input_tensor, target_class):
    model.zero_grad()
    raw_out = model(input_tensor)
    logits = get_class_logits(raw_out)
    if target_class is None:
        target_class = logits.argmax(dim=1).item()
        
    score = logits[0, target_class]
    score.backward(retain_graph=True)
    
    acts = extractor.activations[0].cpu().data.numpy()
    grads = extractor.gradients[0].cpu().data.numpy()
    
    grads_power_2 = grads ** 2
    grads_power_3 = grads ** 3
    sum_activations = np.sum(acts, axis=(1, 2), keepdims=True)
    
    eps = 1e-7
    aij = grads_power_2 / (2 * grads_power_2 + sum_activations * grads_power_3 + eps)
    aij = np.where(grads != 0, aij, 0)
    
    weights = np.maximum(grads, 0) * aij
    weights = np.sum(weights, axis=(1, 2))
    
    cam = np.sum(weights[:, None, None] * acts, axis=0)
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    if np.max(cam) > 0:
        cam = cam / np.max(cam)
    return cam, target_class, F.softmax(logits, dim=1)[0].detach().cpu().numpy()

total_start_t = time.time()

for model_idx, model_path in enumerate(model_files):
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    
    print(f"=====================================================================")
    print(f"[{model_idx+1}/{len(model_files)}] Running Grad-CAM++ for Model: {model_name}")
    print(f"=====================================================================")
    
    model_out_dir = os.path.join(RESULTS_DIR, model_name)
    os.makedirs(model_out_dir, exist_ok=True)
    
    # Load state dict first to infer architecture
    state_dict_raw = torch.load(model_path, map_location=device)
    if isinstance(state_dict_raw, dict) and 'model_state_dict' in state_dict_raw:
        state_dict = state_dict_raw['model_state_dict']
    elif isinstance(state_dict_raw, dict) and 'state_dict' in state_dict_raw:
        state_dict = state_dict_raw['state_dict']
    else:
        state_dict = state_dict_raw
        
    arch = infer_arch(state_dict)
    
    model = get_model(arch, num_classes=7, pretrained=False)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    target_layer = resolve_target_layer(model, arch)
    extractor = FeatureExtractor(model, target_layer)
    
    print(f"Architecture detected: {arch} | Layer attached: {target_layer.__class__.__name__}")
    
    start_t = time.time()
    for i, img_path in enumerate(image_files):
        img_filename = os.path.basename(img_path)
        img_id = os.path.splitext(img_filename)[0]
        
        orig_pil = Image.open(img_path).convert('RGB')
        orig_img_np = np.array(orig_pil)
        orig_img_cv = cv2.resize(cv2.cvtColor(orig_img_np, cv2.COLOR_RGB2BGR), (224, 224))
        
        input_tensor = preprocess(orig_pil).unsqueeze(0).to(device)
        
        cam, pred_idx, probs = compute_gradcam_plus_plus(model, extractor, input_tensor, target_class=None)
        pred_code = CLASS_NAMES[pred_idx]
        conf = probs[pred_idx] * 100
        
        print(f"  [{i+1}/{len(image_files)}] {img_id} -> Pred: {pred_code.upper()} ({conf:.1f}%)")
        
        overlay, _ = apply_colormap_on_image(orig_img_cv, cam)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(orig_img_np)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        axes[1].imshow(overlay_rgb)
        axes[1].set_title(f"Grad-CAM++\nPred: {pred_code.upper()} ({conf:.1f}%)")
        axes[1].axis('off')
        
        plt.tight_layout()
        out_path = os.path.join(model_out_dir, f"{img_id}_pred_{pred_code}_GradCAM++.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        raw_npy_path = os.path.join(model_out_dir, f"{img_id}_pred_{pred_code}_GradCAM++_raw.npy")
        np.save(raw_npy_path, cam)

    extractor.remove_hooks()
    print(f"--> Finished {model_name} in {time.time() - start_t:.1f}s!\n")

print(f"All {len(model_files)} models evaluated with Grad-CAM++ in {time.time() - total_start_t:.1f}s!")
print(f"Results saved across {len(model_files)} subfolders in: {RESULTS_DIR}")
