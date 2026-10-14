# ================== app/services/plate_service.py ==================
import cv2
import re
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
from app.services.plate_lprnet import recognize_plate_lprnet
import os
import datetime
import pandas as pd

# ---------------- LOAD MODELS ----------------
plate_model = YOLO("app/models/plate.pt")

ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    show_log=False
)

# ---------------- CONSTANTS ----------------
INDIAN_STATE_CODES = {
    'AP','AR','AS','BR','CG','GA','GJ','HR','HP','JH','KA','KL',
    'MP','MH','MN','ML','MZ','NL','OD','OR','PB','RJ','SK','TN',
    'TS','TR','UP','UK','WB','DL','AN','CH','DD','DN','JK','LA',
    'LD','PY'
}

# ---------------- VALIDATION ----------------
def validate_indian_plate(text):
    """
    Validate Indian license plate format
    Returns: (cleaned_text, is_valid)
    """
    if not text or not isinstance(text, str):
        return None, False

    # Clean and uppercase
    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    
    # Need at least 6 characters
    if len(text) < 6:
        return text, False

    # Pattern 1: Standard format XX00XX0000 (e.g., MH12AB1234)
    p1 = r'^([A-Z]{2})(\d{2})([A-Z]{1,3})(\d{4})$'
    
    # Pattern 2: Bharat series format 00BH0000XX (e.g., 22BH1234AB)
    p2 = r'^(\d{2})(BH)(\d{4})([A-Z]{2})$'

    if re.match(p1, text) and text[:2] in INDIAN_STATE_CODES:
        return text, True
    if re.match(p2, text):
        return text, True

    # Return text but mark as invalid
    return text, False

# ---------------- PREPROCESS (YELLOW PLATES FIX) ----------------
def preprocess_plate(bgr):
    """Enhance plate image for better OCR"""
    if bgr is None or bgr.size == 0:
        return None
    
    try:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Reduce yellow saturation
        s = cv2.normalize(s, None, 0, 120, cv2.NORM_MINMAX)

        hsv = cv2.merge([h, s, v])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(2.0, (8, 8))
        gray = clahe.apply(gray)

        return gray
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return None


def detect_best_plate(img):
    """Detect all plate bounding boxes using YOLO"""
    try:
        #img = cv2.convertScaleAbs(img, alpha=1.2, beta=-20)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)

        lab = cv2.merge((l,a,b))
        img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        result = plate_model(img_clahe)[0]

        all_boxes = []

        for box in result.boxes:
            conf = float(box.conf[0])
            coords = box.xyxy[0].cpu().numpy().tolist()
            x1, y1, x2, y2 = map(int, coords)
            all_boxes.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf
            })

        return all_boxes
    
    except Exception as e:
        print(f"Plate detection error: {e}")
        return []

# ---------------- MULTI-LINE TEXT EXTRACTION ----------------
def extract_multiline_text(ocr_result, plate_height):
    """
    Extract and combine multi-line text from PaddleOCR results
    
    Args:
        ocr_result: PaddleOCR result list
        plate_height: Height of the plate crop for line separation
        
    Returns:
        (combined_text, average_confidence)
    """
    if not ocr_result or not ocr_result[0]:
        return None, 0.0
    
    # Collect all detections
    detections = []
    for line in ocr_result[0]:
        bbox = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        text_info = line[1]  # (text, confidence)
        text = text_info[0]
        conf = text_info[1]
        
        # Calculate center y-coordinate
        y_coords = [point[1] for point in bbox]
        y_center = sum(y_coords) / len(y_coords)
        
        # Calculate leftmost x-coordinate
        x_coords = [point[0] for point in bbox]
        x_left = min(x_coords)
        
        detections.append({
            'text': text.strip(),
            'conf': conf,
            'y_center': y_center,
            'x_left': x_left
        })
    
    if not detections:
        return None, 0.0
    
    # Separate into lines based on y-coordinate
    # Use half the plate height as threshold
    threshold = plate_height / 2
    top_line = []
    bottom_line = []
    
    for det in detections:
        if det['y_center'] < threshold:
            top_line.append(det)
        else:
            bottom_line.append(det)
    
    # Sort each line by x-coordinate (left to right)
    top_line.sort(key=lambda x: x['x_left'])
    bottom_line.sort(key=lambda x: x['x_left'])
    
    # Combine texts
    all_texts = []
    all_confs = []
    
    # Add top line
    if top_line:
        top_text = ''.join([d['text'] for d in top_line])
        all_texts.append(top_text)
        all_confs.extend([d['conf'] for d in top_line])
    
    # Add bottom line
    if bottom_line:
        bottom_text = ''.join([d['text'] for d in bottom_line])
        all_texts.append(bottom_text)
        all_confs.extend([d['conf'] for d in bottom_line])
    
    # Combine with space
    combined_text = ' '.join(all_texts) if all_texts else None
    avg_conf = sum(all_confs) / len(all_confs) if all_confs else 0.0
    
    return combined_text, avg_conf

# ---------------- OCR PIPELINE ----------------
def read_plate(plate_crop):
    """
    Multi-method OCR pipeline with multi-line support:
    1️⃣ LPRNet (fast, specialized)
    2️⃣ PaddleOCR with multi-line extraction (robust fallback)
    
    Returns: (text, confidence, is_valid)
    """
    if plate_crop is None or plate_crop.size == 0:
        return None, 0.0, False

    plate_h, plate_w = plate_crop.shape[:2]

    # ---------- 1️⃣ TRY LPRNet ----------
    try:
        text = recognize_plate_lprnet(plate_crop)
        if text:  # LPRNet returned something
            text, valid = validate_indian_plate(text)
            if valid:
                print(f"✓ LPRNet recognized: {text}")
                return text, 0.95, True
            else:
                print(f"⚠ LPRNet output invalid: {text}")
    except Exception as e:
        print(f"LPRNet error: {e}")

    # ---------- 2️⃣ FALLBACK TO PaddleOCR with MULTI-LINE SUPPORT ----------
    try:
        processed = preprocess_plate(plate_crop)
        if processed is None:
            return None, 0.0, False
            
        rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)

        # Get OCR results - this returns ALL text regions
        ocr_result = ocr.ocr(rgb, cls=True)
        
        if ocr_result and ocr_result[0]:
            # Extract multi-line text
            text, conf = extract_multiline_text(ocr_result, plate_h)
            
            if text:
                # Clean and validate
                text, valid = validate_indian_plate(text)
                
                if valid:
                    print(f"✓ PaddleOCR recognized: {text} (conf: {conf:.2f})")
                else:
                    print(f"⚠ PaddleOCR output invalid: {text}")
                    
                return text, float(conf), valid
            
    except Exception as e:
        print(f"PaddleOCR error: {e}")

    return None, 0.0, False

# ---------------- ALTERNATIVE: Try Original Image Too ----------------
def read_plate_enhanced(plate_crop):
    """
    Enhanced version that tries both original and preprocessed images
    for better multi-line detection
    """
    if plate_crop is None or plate_crop.size == 0:
        return None, 0.0, False

    plate_h, plate_w = plate_crop.shape[:2]
    
    # ---------- 1️⃣ TRY LPRNet ----------
    try:
        text = recognize_plate_lprnet(plate_crop)
        if text:
            text, valid = validate_indian_plate(text)
            if valid:
                print(f"✓ LPRNet recognized: {text}")
                return text, 0.95, True
    except Exception as e:
        print(f"LPRNet error: {e}")

    # ---------- 2️⃣ TRY PADDLEOCR ON ORIGINAL ----------
    best_text = None
    best_conf = 0.0
    best_valid = False
    
    # Try 1: Original color image
    try:
        ocr_result = ocr.ocr(plate_crop, cls=True)
        if ocr_result and ocr_result[0]:
            text, conf = extract_multiline_text(ocr_result, plate_h)
            if text:
                text, valid = validate_indian_plate(text)
                if valid or conf > best_conf:
                    best_text = text
                    best_conf = conf
                    best_valid = valid
                    if valid:
                        print(f"✓ PaddleOCR (original) recognized: {text}")
    except Exception as e:
        print(f"PaddleOCR (original) error: {e}")
    
    # Try 2: Preprocessed image
    try:
        processed = preprocess_plate(plate_crop)
        if processed is not None:
            rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
            ocr_result = ocr.ocr(rgb, cls=True)
            
            if ocr_result and ocr_result[0]:
                text, conf = extract_multiline_text(ocr_result, plate_h)
                if text:
                    text, valid = validate_indian_plate(text)
                    if valid or (not best_valid and conf > best_conf):
                        best_text = text
                        best_conf = conf
                        best_valid = valid
                        if valid:
                            print(f"✓ PaddleOCR (preprocessed) recognized: {text}")
    except Exception as e:
        print(f"PaddleOCR (preprocessed) error: {e}")
    
    if best_text:
        return best_text, best_conf, best_valid
    
    return None, 0.0, False

# ---------------- MAIN ANALYSIS ----------------
def analyze_plate(img):
    """
    Multi-plate ANPR pipeline with multi-line plate support
    
    Input: BGR image
    Output: list of detection dicts with keys:
            - bbox: [x1, y1, x2, y2]
            - text: recognized plate text
            - confidence: OCR confidence
            - valid: whether format is valid
    """
    if img is None or img.size == 0:
        return []

    results = []

    # ---------------- STEP 1: YOLO DETECTION ----------------
    detected_boxes = detect_best_plate(img)

    if not detected_boxes:
        return results

    h, w = img.shape[:2]

    for det in detected_boxes:
        try:
            x1, y1, x2, y2 = det["bbox"]
            
            # Add padding for better OCR
            pad = 5
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)

            crop = img[y1:y2, x1:x2]

            if crop.size > 0:
                # Use enhanced OCR for multi-line detection
                text, conf, valid = read_plate_enhanced(crop)

                results.append({
                    "bbox": [x1, y1, x2, y2],
                    "text": text if text else "Unknown",
                    "confidence": conf,
                    "valid": valid
                })
        except Exception as e:
            print(f"⚠ Error processing plate crop: {e}")

    return results



