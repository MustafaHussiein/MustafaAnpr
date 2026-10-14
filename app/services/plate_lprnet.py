import torch
import torch.nn as nn
import cv2
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Indian/Latin character set: 36 characters + 1 CTC blank = 37 classes
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NUM_CHARS = 37


class SmallBasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        ch = out_ch // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, (3, 1), padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, (1, 3), padding=(0, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, out_ch, 1),
        )

    def forward(self, x):
        return self.block(x)


class LPRNet(nn.Module):
    def __init__(self, num_classes=NUM_CHARS):
        super().__init__()
        
        self.backbone = nn.Sequential(
            # Layer 0-3: Initial conv block
            nn.Conv2d(3, 64, 3, stride=1, padding=1),              # 0
            nn.BatchNorm2d(64),                                     # 1
            nn.ReLU(inplace=True),                                  # 2
            nn.MaxPool2d(3, stride=1, padding=1),                  # 3
            
            # Layer 4-7: Block 1 (64->128)
            SmallBasicBlock(64, 128),                               # 4
            nn.BatchNorm2d(128),                                    # 5
            nn.ReLU(inplace=True),                                  # 6
            nn.MaxPool2d(3, stride=(2, 1), padding=1),             # 7
            
            # Layer 8-10: Block 2 (64->256)
            SmallBasicBlock(64, 256),                               # 8
            nn.BatchNorm2d(256),                                    # 9
            nn.ReLU(inplace=True),                                  # 10
            
            # Layer 11-15: Block 3 (256->256)
            SmallBasicBlock(256, 256),                              # 11
            nn.BatchNorm2d(256),                                    # 12
            nn.ReLU(inplace=True),                                  # 13
            nn.MaxPool2d((3, 3), stride=(1, 1), padding=(1, 1)),   # 14
            nn.Dropout(0.5),                                        # 15
            
            # Layer 16-19: Conv block
            nn.Conv2d(64, 256, (1, 4)),                            # 16
            nn.BatchNorm2d(256),                                    # 17
            nn.ReLU(inplace=True),                                  # 18
            nn.Dropout(0.5),                                        # 19
            
            # Layer 20-21: Output - ADAPTIVE KERNEL SIZE
            nn.Conv2d(256, num_classes, (1, 1)),  # Changed from (13,1) to (1,1)
            nn.BatchNorm2d(num_classes),                            # 21
        )
        
        # Container: 485 = 64+128+256+37
        self.container = nn.Sequential(
            nn.Conv2d(485, num_classes, 1)
        )

    def forward(self, x):
        # Get input dimensions
        b, c, h, w = x.shape
        
        # Initial conv (0-3)
        for i in range(4):
            x = self.backbone[i](x)
        
        # Block 1 (4-7)
        for i in range(4, 8):
            x = self.backbone[i](x)
        
        # Block 2 (8-10) - expects 64 channels
        x_small = x[:, :64, :, :]
        for i in range(8, 11):
            if i == 8:
                x = self.backbone[i](x_small)
            else:
                x = self.backbone[i](x)
        
        # Block 3 (11-15)
        for i in range(11, 16):
            x = self.backbone[i](x)
        
        # Conv block (16-19) - expects 64 channels
        x_small = x[:, :64, :, :]
        for i in range(16, 20):
            if i == 16:
                x = self.backbone[i](x_small)
            else:
                x = self.backbone[i](x)
        
        # Output (20-21) - Use adaptive pooling if height is too small
        current_h = x.shape[2]
        
        # Apply global average pooling over height dimension instead of conv
        # This makes it work regardless of input height
        x = torch.mean(x, dim=2, keepdim=True)  # (B, 256, 1, W)
        
        # Now apply 1x1 conv instead of 13x1
        x = self.backbone[20](x)
        x = self.backbone[21](x)
        
        # Prepare for CTC: (B, C, H, W) -> (W, B, C)
        x = x.squeeze(2)  # Remove height dimension -> (B, C, W)
        x = x.permute(2, 0, 1)  # Reorder to (W, B, C) for CTC
        
        return x


# ============ LOAD MODEL ============
print("Loading Indian License Plate Recognition Model...")
print(f"Character set: {CHARS}")
print(f"Total classes: {NUM_CHARS}")

model = LPRNet(num_classes=NUM_CHARS)

try:
    state_dict = torch.load("app/models/lprnet.pth", map_location=DEVICE, weights_only=False)
    
    # Load weights, ignoring size mismatches for layer 20 (we changed kernel size)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() 
                      if k in model_dict and v.shape == model_dict[k].shape}
    
    # Update model dict with matched weights
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    loaded = len(pretrained_dict)
    total = len(model_dict)
    print(f"✓ Loaded {loaded}/{total} parameters ({loaded*100//total}%)")
    
    # List what wasn't loaded
    not_loaded = set(model_dict.keys()) - set(pretrained_dict.keys())
    if not_loaded:
        print(f"⚠ Randomly initialized: {', '.join(sorted(not_loaded))}")
    
except FileNotFoundError:
    print("✗ ERROR: Model file not found at app/models/lprnet.pth")
    print("  Please download from: https://github.com/sanchit2843/Indian_LPR")
except Exception as e:
    print(f"✗ ERROR loading model: {e}")

model.to(DEVICE)
model.eval()


# ============ PREPROCESSING ============
def preprocess_plate(img):
    """Preprocess license plate image"""
    # Resize to model input size
    img = cv2.resize(img, (94, 24))
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize to [-1, 1]
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    
    # Convert to CHW format
    img = np.transpose(img, (2, 0, 1))
    
    return torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(DEVICE)


# ============ CTC DECODING ============
def ctc_decode(preds):
    """Decode CTC predictions to text"""
    # Get most likely character at each timestep
    preds = preds.argmax(dim=2).cpu().numpy()
    
    results = []
    for pred in preds.T:  # Iterate over batch
        chars = []
        prev = -1
        
        for p in pred:
            # Skip blank (0) and consecutive duplicates
            if p != 0 and p != prev:
                if 1 <= p <= len(CHARS):
                    chars.append(CHARS[p - 1])
            prev = p
        
        results.append(''.join(chars))
    
    return results


# ============ MAIN API FUNCTION ============
def recognize_plate_lprnet(plate_img):
    """
    Recognize Indian license plate from image
    
    Args:
        plate_img: OpenCV BGR image of the license plate
        
    Returns:
        str: Recognized plate text (e.g., "MH12AB1234") or None if failed
    """
    if plate_img is None or plate_img.size == 0:
        return None
    
    try:
        # Preprocess
        tensor = preprocess_plate(plate_img)
        
        # Inference
        with torch.no_grad():
            preds = model(tensor)
        
        # Decode
        plates = ctc_decode(preds)
        plate_text = plates[0] if plates else ""
        
        # Indian plates are typically 10 characters (e.g., MH12AB1234)
        # But return anything with 6+ characters
        return plate_text if len(plate_text) >= 6 else None
        
    except Exception as e:
        print(f"Recognition error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============ TESTING ============
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Model ready for Indian License Plate Recognition!")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Character set: {CHARS}")
    print(f"Expected plate format: XX00XX0000 (e.g., MH12AB1234)")
    print("\nUsage:")
    print("  plate_text = recognize_plate_lprnet(plate_image)")