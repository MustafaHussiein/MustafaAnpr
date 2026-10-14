# ================== app/api/v1/anpr.py ==================
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import cv2
import numpy as np
import os
import shutil
import uuid
import traceback
from app.services.plate_service import analyze_plate
import pandas as pd
from datetime import datetime

router = APIRouter(prefix="/anpr", tags=["ANPR"])

# Output directory for processed images
OUTPUT_DIR = "outputImgs"
# --- LOGGING CONFIGURATION ---
LOG_DIR = "app/data"
CSV_PATH = os.path.join(LOG_DIR, "anpr_history.csv")
XLSX_PATH = os.path.join(LOG_DIR, "anpr_history.xlsx")
os.makedirs(LOG_DIR, exist_ok=True)

def log_detection_to_files(plate_text, image_url):
    """Saves the plate data and its public URL to CSV and Excel."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = {
        "Timestamp": [timestamp],
        "Plate_Text": [plate_text],
        "Image_URL": [image_url]
    }
    df_new = pd.DataFrame(new_entry)

    # Save to CSV (Append)
    df_new.to_csv(CSV_PATH, mode='a', index=False, header=not os.path.exists(CSV_PATH))

    # Save to Excel (Append/Merge)
    if os.path.exists(XLSX_PATH):
        try:
            df_old = pd.read_excel(XLSX_PATH)
            df_final = pd.concat([df_old, df_new], ignore_index=True)
            df_final.to_excel(XLSX_PATH, index=False)
        except Exception as e:
            print(f"Excel error: {e}")
            df_new.to_excel(XLSX_PATH, index=False)
    else:
        df_new.to_excel(XLSX_PATH, index=False)

@router.get("/search-csv")
async def search_csv_records(start: str = None, end: str = None, plate: str = None):
    if not os.path.exists(CSV_PATH):
        return {"results": []}

    try:
        # Read the generated CSV
        df = pd.read_csv(CSV_PATH)
        
        # Ensure Timestamp is a datetime object for comparison
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        # Filter by Date Range
        if start:
            df = df[df['Timestamp'] >= pd.to_datetime(start)]
        if end:
            df = df[df['Timestamp'] <= pd.to_datetime(end)]
        
        # Filter by Plate (Partial match, case-insensitive)
        if plate:
            df = df[df['Plate_Text'].str.contains(plate, case=False, na=False)]

        # Sort by newest first
        df = df.sort_values(by='Timestamp', ascending=False)

        # Convert back to string and rename columns for JS frontend
        df['Timestamp'] = df['Timestamp'].dt.strftime("%Y-%m-%d %H:%M:%S")
        
        results = df.rename(columns={
            "Timestamp": "timestamp", 
            "Plate_Text": "plate", 
            "Image_URL": "image_url"
        }).to_dict(orient="records")

        return {"results": results}
    except Exception as e:
        print(f"Search error: {e}")
        return {"results": [], "error": str(e)}
    
def cleanup_output_dir():
    """Ensure directory exists without deleting existing history"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

def draw_plate_detections(img, plates):
    """Draw only plate detections on image"""
    annotated = img.copy()
    
    PLATE_COLOR = (255, 0, 0)       # Blue
    TEXT_COLOR = (255, 255, 255)    # White
    BG_COLOR = (0, 0, 0)            # Black
    
    img_h, img_w = img.shape[:2]
    
    for idx, plate in enumerate(plates):
        try:
            bbox = plate.get("bbox", [])
            if len(bbox) != 4:
                continue
                
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(x1, img_w))
            y1 = max(0, min(y1, img_h))
            x2 = max(0, min(x2, img_w))
            y2 = max(0, min(y2, img_h))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            plate_text = plate.get("text", "Unknown")
            conf = plate.get("confidence", 0.0)
            
            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), PLATE_COLOR, 2)
            
            # Label
            label = f"Plate: {plate_text} ({conf:.2f})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Label background
            label_y1 = max(y1 - label_h - 8, 0)
            label_y2 = max(y1, label_h + 8)
            cv2.rectangle(annotated, (x1, label_y1), (x1 + label_w + 8, label_y2), PLATE_COLOR, -1)
            
            # Label text
            text_y = max(y1 - 4, label_h + 4)
            cv2.putText(annotated, label, (x1 + 4, text_y), 
                        font, font_scale, TEXT_COLOR, thickness, cv2.LINE_AA)
        except Exception as e:
            print(f"⚠ Error drawing plate {idx}: {e}")
    
    # Summary
    summary = f"Plates: {len(plates)}"
    cv2.rectangle(annotated, (10, 10), (200, 50), BG_COLOR, -1)
    cv2.putText(annotated, summary, (20, 38), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2, cv2.LINE_AA)
    
    return annotated


# ============================================================
# SEPARATE ENDPOINTS FOR EACH SERVICE
# ============================================================
@router.post("/analyze-plates")
async def analyze_plates_only(image: UploadFile = File(...)):
    try:
        cleanup_output_dir()
        
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        plates = analyze_plate(img)
        
        for plate in plates:
            try:
                bbox = plate.get("bbox", [])
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    crop = img[y1:y2, x1:x2]
                    
                    if crop.size > 0:
                        crop_filename = f"crop_{uuid.uuid4().hex[:6]}.jpg"
                        crop_path = os.path.join(OUTPUT_DIR, crop_filename)
                        cv2.imwrite(crop_path, crop)
                        
                        # CONSTRUCT THE URL
                        public_url = f"/anpr/download/{crop_filename}"
                        plate["crop_url"] = public_url
                        
                        # SAVE TO CSV/EXCEL IMMEDIATELY
                        log_detection_to_files(plate.get("text", "Unknown"), public_url)
                        
            except Exception as e:
                print(f"⚠ Failed to create crop or log: {e}")

        # ... (rest of your existing drawing and full image save logic)
        
        return {
            "plates": plates,
            "annotated_image_url": f"/anpr/download/{crop_filename}",
            "summary": {"total_plates": len(plates)}
        }
    
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download/{filename}")
async def download_annotated_image(filename: str):
    """Download annotated image"""
    # Security validation
    if not filename.endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    if '..' in filename or '/' in filename or '\\' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(filepath, media_type="image/jpeg", filename=filename)