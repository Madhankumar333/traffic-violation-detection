from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import cv2
import os
from ultralytics import YOLO
from datetime import datetime
import uuid
import numpy as np
from PIL import Image
import torch
from google import genai
import tempfile
import shutil
import traceback
from sort import Sort
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import base64
import math
import copy

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'violations'
app.config['TEMP_FOLDER'] = 'temp_videos'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['TEMP_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['OUTPUT_FOLDER'], 'images'), exist_ok=True)

# Email Configuration (use environment variables for security)
EMAIL_ADDRESS = os.environ.get("EMAIL_ADDRESS", "ayushtiwari.creatorslab@gmail.com")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "tecx bcym vxdz dtni")
RECIPIENT_EMAIL = os.environ.get("RECIPIENT_EMAIL", "tn.bender2005@gmail.com")

# Gemini API Key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyCMY5-lrX7kPKBesjgQpm9O1bvO3jV65Io")
try:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
except:
    gemini_client = None
    print("Gemini client not initialized")

# Model paths
HELMET_MODEL_PATH = "best_helmetdetection.pt"
TRIPLE_MODEL_PATH = "triple_riding.pt"
YOLOV8_MODEL_PATH = "yolov8n.pt"

# Load models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Loading models on device: {device}")

try:
    helmet_model = YOLO(HELMET_MODEL_PATH).to(device)
    print(f"✓ Helmet model loaded")
except Exception as e:
    print(f"✗ Error loading helmet model: {e}")
    helmet_model = None

try:
    triple_model = YOLO(TRIPLE_MODEL_PATH).to(device)
    print(f"✓ Triple riding model loaded")
except Exception as e:
    print(f"✗ Error loading triple model: {e}")
    triple_model = None

try:
    yolo_model = YOLO(YOLOV8_MODEL_PATH).to(device)
    print(f"✓ YOLOv8n model loaded for red light detection")
except Exception as e:
    print(f"✗ Error loading YOLOv8n model: {e}")
    yolo_model = None

violations_db = []

FINE_AMOUNTS = {
    'NO_HELMET': 1000,
    'TRIPLE_RIDING': 2000,
    'RED_LIGHT': 5000
}

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# ── UNCHANGED ─────────────────────────────────────────────────────────────────
def is_red_light(frame, light_region):
    try:
        x1, y1, x2, y2 = light_region
        x1, x2 = max(0, min(x1, x2)), max(x1, x2)
        y1, y2 = max(0, min(y1, y2)), max(y1, y2)
        height, width = frame.shape[:2]
        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))
        if x1 >= x2 or y1 >= y2:
            print(f"  Invalid light region coordinates: {x1},{y1},{x2},{y2}")
            return False
        light_crop = frame[y1:y2, x1:x2]
        if light_crop.size == 0:
            print(f"  Empty light crop")
            return False
        hsv = cv2.cvtColor(light_crop, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        red_pixels = cv2.countNonZero(mask)
        total_pixels = light_crop.shape[0] * light_crop.shape[1]
        red_percentage = (red_pixels / total_pixels) * 100
        print(f"  Red percentage: {red_percentage:.2f}%")
        return red_percentage > 3
    except Exception as e:
        print(f"  Error in red light detection: {e}")
        traceback.print_exc()
        return False


def send_challan_email(challan_data):
    try:
        msg = MIMEMultipart('alternative')
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = RECIPIENT_EMAIL
        msg['Subject'] = f"Traffic Challan - {challan_data['challan_id']} - {challan_data['plate_number']}"
        violations_list = "<br>".join([f"• {v['type'].replace('_', ' ')}: ₹{FINE_AMOUNTS[v['type']]}"
                                       for v in challan_data['violations']])
        total_fine = sum([FINE_AMOUNTS[v['type']] for v in challan_data['violations']])
        html = f"""
        <html><body style="font-family: Arial, sans-serif;">
            <h2 style="color: #d32f2f;">Traffic Violation Challan</h2><hr>
            <table style="width: 100%; border-collapse: collapse;">
                <tr><td style="padding: 8px; font-weight: bold;">Challan ID:</td>
                    <td style="padding: 8px;">{challan_data['challan_id']}</td></tr>
                <tr><td style="padding: 8px; font-weight: bold;">Vehicle Number:</td>
                    <td style="padding: 8px; color: #d32f2f; font-size: 1.2em; font-weight: bold;">{challan_data['plate_number']}</td></tr>
                <tr><td style="padding: 8px; font-weight: bold;">Tracked Vehicle ID:</td>
                    <td style="padding: 8px;">{challan_data.get('tracked_id', 'N/A')}</td></tr>
                <tr><td style="padding: 8px; font-weight: bold;">Date & Time:</td>
                    <td style="padding: 8px;">{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</td></tr>
                <tr><td style="padding: 8px; font-weight: bold;">Total Violations:</td>
                    <td style="padding: 8px;">{challan_data['total_violations']}</td></tr>
            </table>
            <h3 style="color: #d32f2f;">Violations Detected:</h3>
            <div style="background-color: #ffebee; padding: 15px; border-radius: 5px;">{violations_list}</div>
            <h3 style="color: #388e3c;">Total Fine Amount: ₹{total_fine}</h3>
            <p><small>Auto-generated challan. Evidence images attached.</small></p>
            <hr><p style="color: #666; font-size: 12px;">Issued by Traffic Regulations Authority<br>
            Powered by AI Traffic Violation Detection System</p>
        </body></html>"""
        msg.attach(MIMEText(html, 'html'))
        for violation in challan_data['violations']:
            violation_image_path = os.path.join(app.config['OUTPUT_FOLDER'], 'images', violation['image'])
            if os.path.exists(violation_image_path):
                with open(violation_image_path, 'rb') as f:
                    img = MIMEImage(f.read())
                    img.add_header('Content-Disposition', 'attachment',
                                   filename=f"{violation['type']}_{violation['id']}.jpg")
                    msg.attach(img)
            if 'plate_image' in violation and violation['plate_image']:
                plate_image_path = os.path.join(app.config['OUTPUT_FOLDER'], 'images', violation['plate_image'])
                if os.path.exists(plate_image_path):
                    with open(plate_image_path, 'rb') as f:
                        img = MIMEImage(f.read())
                        img.add_header('Content-Disposition', 'attachment',
                                       filename=f"plate_{violation['id']}.jpg")
                        msg.attach(img)
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        print(f"✓ Challan email sent for Vehicle ID: {challan_data.get('tracked_id')} - Plate: {challan_data['plate_number']}")
        return True
    except Exception as e:
        print(f"✗ Email sending failed: {e}")
        traceback.print_exc()
        return False


def extract_plate_with_gemini(image_path):
    if gemini_client is None:
        return "UNKNOWN"
    try:
        my_file = gemini_client.files.upload(file=image_path)
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[my_file, "Extract only the vehicle number plate text from this image. Return only the plate number, nothing else."],
        )
        plate_text = response.text.strip()
        return plate_text if plate_text else "UNKNOWN"
    except Exception as e:
        print(f"Gemini API error: {e}")
        return "UNKNOWN"


def run_model_with_class_confidence(model, input_data, class_confidences, verbose=False):
    if model is None:
        return []
    min_conf = min(class_confidences.values())
    results = model(input_data, conf=min_conf, verbose=verbose)
    filtered_results = []
    for r in results:
        if r.boxes is not None:
            filtered_boxes = []
            for box in r.boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                conf = float(box.conf[0])
                required_conf = class_confidences.get(class_name, min_conf)
                if conf >= required_conf:
                    filtered_boxes.append(box)
            if filtered_boxes:
                filtered_result = copy.deepcopy(r)
                filtered_result.boxes.data = torch.stack([box.data[0] for box in filtered_boxes])
                filtered_result.boxes.orig_shape = r.boxes.orig_shape
                filtered_results.append(filtered_result)
            else:
                filtered_result = copy.deepcopy(r)
                filtered_result.boxes = None
                filtered_results.append(filtered_result)
        else:
            filtered_results.append(r)
    return filtered_results


def calculate_overlap_ratio(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    xi1 = max(x1, x3)
    yi1 = max(y1, y3)
    xi2 = min(x2, x4)
    yi2 = min(y2, y4)
    if xi1 >= xi2 or yi1 >= yi2:
        return 0.0
    intersection = (xi2 - xi1) * (yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    if box1_area <= 0:
        return 0.0
    return intersection / box1_area


def calculate_plate_clarity_score(plate_image):
    try:
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = np.mean(gray)
        clarity_score = (laplacian_var / 100.0) * (brightness / 255.0)
        return clarity_score
    except:
        return 0.0


def process_motorcycle(motorcycle_box, frame, temp_dir, frame_number, tracked_id):
    x1, y1, x2, y2 = map(int, motorcycle_box.xyxy[0])
    motorcycle_crop = frame[y1:y2, x1:x2]
    if motorcycle_crop.size == 0:
        print(f"  Empty motorcycle crop for ID {tracked_id}")
        return {'violations': [], 'plate': 'UNKNOWN', 'num_people': 0,
                'num_helmets': 0, 'num_faces': 0, 'best_plate_image': None,
                'bbox': [x1, y1, x2, y2], 'tracked_id': tracked_id}
    violations = []
    helmet_confidences = {'helmet': 0.75, 'motorcyclist': 0.55, 'motorcycle': 0.55,
                          'license_plate': 0.3, 'person': 0.6, 'no-helmet': 0.6}
    helmet_results = run_model_with_class_confidence(helmet_model, motorcycle_crop, helmet_confidences)
    helmets = []
    license_plates = []
    people = []
    for r in helmet_results:
        if r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                class_name = helmet_model.names[cls_id]
                conf = float(box.conf[0])
                bbox = box.xyxy[0].tolist()
                if class_name in ['helmet']:
                    helmets.append({'bbox': bbox, 'conf': conf})
                elif class_name in ['license_plate', 'license-plate']:
                    license_plates.append({'bbox': bbox, 'conf': conf})
                elif class_name in ['person', 'no-helmet']:
                    people.append({'bbox': bbox, 'conf': conf})
    adjusted_people = []
    for person in people:
        person_bbox = person['bbox']
        overlap_found = False
        for helmet in helmets:
            if calculate_overlap_ratio(person_bbox, helmet['bbox']) > 0.5:
                overlap_found = True
                break
        if not overlap_found:
            adjusted_people.append(person)
    num_faces = len(adjusted_people)
    num_helmets = len(helmets)
    if num_helmets == 0 or num_faces > 0:
        violations.append('NO_HELMET')
    if triple_model:
        triple_confidences = {'motorbike': 0.5, 'motorcycle': 0.5, 'triple': 0.5, 'triple_riding': 0.5}
        triple_results = run_model_with_class_confidence(triple_model, motorcycle_crop, triple_confidences)
        for t_result in triple_results:
            if t_result.boxes is not None:
                for t_box in t_result.boxes:
                    t_label = t_result.names[int(t_box.cls[0])]
                    if 'triple' in t_label.lower():
                        violations.append('TRIPLE_RIDING')
                        break
    best_plate_image = None
    best_plate_score = 0.0
    license_plate_text = "UNKNOWN"
    if license_plates:
        for lp in license_plates:
            lp_bbox = lp['bbox']
            x1_lp, y1_lp, x2_lp, y2_lp = map(int, lp_bbox)
            if x1_lp >= x2_lp or y1_lp >= y2_lp:
                continue
            h, w = motorcycle_crop.shape[:2]
            x1_lp = max(0, min(x1_lp, w))
            x2_lp = max(0, min(x2_lp, w))
            y1_lp = max(0, min(y1_lp, h))
            y2_lp = max(0, min(y2_lp, h))
            if x1_lp >= x2_lp or y1_lp >= y2_lp:
                continue
            plate_crop = motorcycle_crop[y1_lp:y2_lp, x1_lp:x2_lp]
            if plate_crop.size == 0:
                continue
            clarity_score = calculate_plate_clarity_score(plate_crop)
            if clarity_score > best_plate_score:
                best_plate_score = clarity_score
                best_plate_image = plate_crop
        if best_plate_image is not None and violations:
            lp_path = os.path.join(temp_dir, f"plate_{tracked_id}_{frame_number}.jpg")
            cv2.imwrite(lp_path, best_plate_image)
            license_plate_text = extract_plate_with_gemini(lp_path)
    return {'violations': violations, 'plate': license_plate_text,
            'num_people': num_faces + num_helmets, 'num_helmets': num_helmets,
            'num_faces': num_faces, 'best_plate_image': best_plate_image,
            'bbox': [x1, y1, x2, y2], 'tracked_id': tracked_id}


def process_video_with_tracking(video_path):
    if helmet_model is None:
        raise Exception("Helmet model not loaded")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Could not open video file")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video info: {total_frames} frames, {fps} FPS")
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    processed_ids = {}
    frame_interval = max(1, fps // 8)
    print(f"Frame interval: {frame_interval}")
    temp_dir = tempfile.mkdtemp()
    print(f"Temp directory: {temp_dir}")
    frame_count = 0
    all_violations_by_id = {}
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                print(f"Processing frame {frame_count}/{total_frames}")
                try:
                    frame_resized = cv2.resize(frame, (426, 240))
                    helmet_confidences = {'helmet': 0.75, 'motorcyclist': 0.55,
                                          'motorcycle': 0.55, 'license_plate': 0.3}
                    helmet_results = run_model_with_class_confidence(helmet_model, frame_resized, helmet_confidences)
                    detections = np.empty((0, 5))
                    for yolo_result in helmet_results:
                        boxes = yolo_result.boxes
                        if boxes is None:
                            continue
                        for box in boxes:
                            cls = int(box.cls[0])
                            label = yolo_result.names[cls]
                            if label.lower() not in ['motorbike', 'motorcycle', 'motorcyclist']:
                                continue
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            scale_x = frame.shape[1] / 426
                            scale_y = frame.shape[0] / 240
                            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                            h, w = frame.shape[:2]
                            x1 = max(0, min(x1, w)); x2 = max(0, min(x2, w))
                            y1 = max(0, min(y1, h)); y2 = max(0, min(y2, h))
                            if x1 >= x2 or y1 >= y2:
                                continue
                            detections = np.vstack((detections, np.array([x1, y1, x2, y2, conf])))
                    tracked_objects = tracker.update(detections)
                    for tracked in tracked_objects:
                        x1, y1, x2, y2, tracked_id = tracked
                        tracked_id = int(tracked_id)
                        if tracked_id in processed_ids:
                            continue
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        h, w = frame.shape[:2]
                        x1 = max(0, min(x1, w)); x2 = max(0, min(x2, w))
                        y1 = max(0, min(y1, h)); y2 = max(0, min(y2, h))
                        if x1 >= x2 or y1 >= y2:
                            continue
                        dummy_box = type('obj', (object,), {'xyxy': [torch.tensor([x1, y1, x2, y2])]})()
                        motorcycle_result = process_motorcycle(dummy_box, frame, temp_dir, frame_count, tracked_id)
                        if motorcycle_result['violations']:
                            processed_ids[tracked_id] = {'plate': motorcycle_result['plate'],
                                                          'violations': motorcycle_result['violations'],
                                                          'frame': frame_count,
                                                          'best_plate': motorcycle_result['best_plate_image']}
                            if tracked_id not in all_violations_by_id:
                                all_violations_by_id[tracked_id] = []
                            for violation_type in motorcycle_result['violations']:
                                violation_id = str(uuid.uuid4())[:8]
                                bbox_frame = frame.copy()
                                x1, y1, x2, y2 = map(int, motorcycle_result['bbox'])
                                color = (0, 0, 255) if violation_type == 'NO_HELMET' else (255, 0, 0)
                                cv2.rectangle(bbox_frame, (x1, y1), (x2, y2), color, 3)
                                cv2.putText(bbox_frame, f"{violation_type} ID:{tracked_id}",
                                            (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                                bbox_path = os.path.join(app.config['OUTPUT_FOLDER'], 'images',
                                                         f'{violation_id}_{violation_type.lower()}.jpg')
                                cv2.imwrite(bbox_path, bbox_frame)
                                plate_path = None
                                if motorcycle_result['best_plate_image'] is not None:
                                    plate_path = os.path.join(app.config['OUTPUT_FOLDER'], 'images',
                                                              f'{violation_id}_plate.jpg')
                                    cv2.imwrite(plate_path, motorcycle_result['best_plate_image'])
                                violation_data = {
                                    'id': violation_id, 'type': violation_type,
                                    'plate': motorcycle_result['plate'], 'frame': frame_count,
                                    'timestamp': datetime.now().isoformat(),
                                    'image': f"{violation_id}_{violation_type.lower()}.jpg",
                                    'plate_image': f"{violation_id}_plate.jpg" if plate_path else None,
                                    'num_people': motorcycle_result['num_people'],
                                    'num_helmets': motorcycle_result['num_helmets'],
                                    'num_faces': motorcycle_result['num_faces'],
                                    'tracked_id': tracked_id
                                }
                                all_violations_by_id[tracked_id].append(violation_data)
                                print(f"  → Violation: {violation_type}, ID: {tracked_id}, Plate: {motorcycle_result['plate']}")
                except Exception as e:
                    print(f"  Error processing frame {frame_count}: {e}")
                    traceback.print_exc()
                    continue
            frame_count += 1
        cap.release()
        print(f"Unique vehicles tracked: {len(all_violations_by_id)}")
    except Exception as e:
        print(f"Error in video processing: {e}")
        traceback.print_exc()
        raise
    finally:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass
    challans = generate_individual_challans(all_violations_by_id)
    return challans


# ── ONLY THIS FUNCTION CHANGED ────────────────────────────────────────────────
def detect_red_light_violation(video_path, line_coords, light_region):
    """
    Detect red light violations.
    Changes from original (red light only, nothing else touched):
      FIX 1 - min_hits=1: tracker registers vehicle on very first detection
      FIX 2 - line_y = midpoint of drawn line: correct horizontal reference
      FIX 3 - prev_cy tracking: catches vehicles that jump over line between frames
      FIX 4 - direction-change + band check: nothing missed even at low FPS
      FIX 5 - plate from Gemini: vehicle crop sent to Gemini on violation
    """
    if yolo_model is None:
        raise Exception("YOLOv8n model not loaded")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Could not open video file")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Red Light Video info: {total_frames} frames, {fps} FPS")

    frame_interval = max(1, fps // 6)
    print(f"Frame interval: {frame_interval}")

    # FIX 1: min_hits=1 so vehicle is tracked immediately on first detection
    tracker = Sort(max_age=20, min_hits=1, iou_threshold=0.3)

    # FIX 2: use midpoint Y of the drawn line as horizontal crossing reference
    lx1, ly1, lx2, ly2 = line_coords
    line_y = (ly1 + ly2) // 2

    totalCount = []
    all_violations_by_id = {}
    frame_count = 0
    processed_frames = 0

    # FIX 3: remember each vehicle's previous centre-Y to catch line jumps
    vehicle_prev_y = {}

    temp_dir = tempfile.mkdtemp()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                processed_frames += 1
                print(f"Processing frame {frame_count}/{total_frames}")

                is_light_red = is_red_light(frame, light_region)
                print(f"  Frame {frame_count}: Light is {'RED' if is_light_red else 'NOT RED'}")

                if is_light_red:
                    results = yolo_model(frame, stream=True, verbose=False)
                    detections = np.empty((0, 5))
                    for r in results:
                        boxes = r.boxes
                        if boxes is None:
                            continue
                        for box in boxes:
                            cls = int(box.cls[0])
                            currentclass = classNames[cls]
                            if currentclass in ["car", "truck", "bus", "motorbike"] and float(box.conf[0]) > 0.4:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                conf = math.ceil((float(box.conf[0]) * 100)) / 100
                                detections = np.vstack((detections, np.array([x1, y1, x2, y2, conf])))

                    print(f"  Detected {len(detections)} vehicles")
                    tracked_objects = tracker.update(detections)
                    print(f"  Tracking {len(tracked_objects)} objects")

                    for tracked in tracked_objects:
                        x1, y1, x2, y2, tracked_id = tracked
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        tracked_id = int(tracked_id)

                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2

                        # FIX 3 & 4: crossing check using previous + current position
                        prev_cy = vehicle_prev_y.get(tracked_id, cy)
                        vehicle_prev_y[tracked_id] = cy

                        crossed = (
                            lx1 <= cx <= lx2 and
                            (
                                (prev_cy <= line_y <= cy) or   # passed through going down
                                (prev_cy >= line_y >= cy) or   # passed through going up
                                abs(cy - line_y) <= 20         # currently on the line band
                            )
                        )

                        if crossed and tracked_id not in totalCount:
                            totalCount.append(tracked_id)
                            print(f"  ✗ RED LIGHT VIOLATION: Vehicle ID {tracked_id} crossed at frame {frame_count}")

                            violation_id = str(uuid.uuid4())[:8]

                            # Draw clean violation frame
                            vframe = frame.copy()
                            cv2.line(vframe, (lx1, ly1), (lx2, ly2), (0, 0, 255), 5)
                            cv2.rectangle(vframe, (light_region[0], light_region[1]),
                                          (light_region[2], light_region[3]), (0, 0, 255), 2)
                            cv2.putText(vframe, "RED LIGHT",
                                        (light_region[0], light_region[1]-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            cv2.rectangle(vframe, (x1, y1), (x2, y2), (0, 0, 255), 4)
                            cv2.putText(vframe, f"RED LIGHT VIOLATION ID:{tracked_id}",
                                        (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

                            violation_path = os.path.join(app.config['OUTPUT_FOLDER'], 'images',
                                                          f'{violation_id}_red_light.jpg')
                            cv2.imwrite(violation_path, vframe)

                            # FIX 5: extract plate from vehicle crop via Gemini
                            plate_text = "UNKNOWN"
                            plate_filename = None
                            h_fr, w_fr = frame.shape[:2]
                            cx1 = max(0, min(x1, w_fr))
                            cx2 = max(0, min(x2, w_fr))
                            cy1 = max(0, min(y1, h_fr))
                            cy2 = max(0, min(y2, h_fr))
                            vehicle_crop = frame[cy1:cy2, cx1:cx2]
                            if vehicle_crop.size > 0:
                                veh_path = os.path.join(temp_dir, f"veh_{tracked_id}_{frame_count}.jpg")
                                cv2.imwrite(veh_path, vehicle_crop)
                                plate_text = extract_plate_with_gemini(veh_path)
                                if plate_text != "UNKNOWN":
                                    plate_filename = f'{violation_id}_plate.jpg'
                                    shutil.copy(veh_path,
                                                os.path.join(app.config['OUTPUT_FOLDER'],
                                                             'images', plate_filename))

                            violation_data = {
                                'id': violation_id,
                                'type': 'RED_LIGHT',
                                'plate': plate_text,
                                'frame': frame_count,
                                'timestamp': datetime.now().isoformat(),
                                'image': f'{violation_id}_red_light.jpg',
                                'plate_image': plate_filename,
                                'tracked_id': tracked_id
                            }

                            if tracked_id not in all_violations_by_id:
                                all_violations_by_id[tracked_id] = []
                            all_violations_by_id[tracked_id].append(violation_data)

                else:
                    # Light not red — keep tracker alive, reset stale Y positions
                    tracker.update(np.empty((0, 5)))
                    vehicle_prev_y.clear()

            frame_count += 1

        cap.release()
        print(f"\n=== Red Light Detection Summary ===")
        print(f"Total frames: {total_frames}")
        print(f"Processed frames: {processed_frames}")
        print(f"Total violations: {len(totalCount)}")
        print(f"Unique vehicles with violations: {len(all_violations_by_id)}")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    challans = generate_individual_challans(all_violations_by_id)
    return challans
# ── END OF CHANGED FUNCTION ───────────────────────────────────────────────────


def generate_individual_challans(violations_by_id):
    challans = []
    for tracked_id, violations in violations_by_id.items():
        challan_id = str(uuid.uuid4())[:12].upper()
        plate_number = "UNKNOWN"
        for v in violations:
            if v['plate'] != "UNKNOWN":
                plate_number = v['plate']
                break
        challan = {
            'challan_id': challan_id,
            'plate_number': plate_number,
            'tracked_id': tracked_id,
            'violations': violations,
            'total_violations': len(violations),
            'generated_at': datetime.now().isoformat()
        }
        send_challan_email(challan)
        challans.append(challan)
        print(f"✓ Generated challan for Vehicle ID {tracked_id}: {challan_id}")
    return challans


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_first_frame', methods=['POST'])
def get_first_frame():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file'}), 400
        file = request.files['video']
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['TEMP_FOLDER'], filename)
        file.save(video_path)
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return jsonify({'error': 'Could not read video'}), 400
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({
            'success': True,
            'frame': frame_base64,
            'video_path': filename,
            'width': frame.shape[1],
            'height': frame.shape[0]
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/upload', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file', 'challans': []}), 400
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No selected file', 'challans': []}), 400
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)
        print(f"Processing video: {filename}")
        challans = process_video_with_tracking(video_path)
        violations_db.extend(challans)
        print(f"Generated {len(challans)} individual challans")
        return jsonify({'success': True, 'challans': challans, 'total_challans': len(challans)})
    except Exception as e:
        error_msg = str(e)
        traceback.print_exc()
        return jsonify({'success': False, 'error': error_msg, 'challans': []}), 500
    finally:
        if 'video_path' in locals() and os.path.exists(video_path):
            try:
                os.unlink(video_path)
            except:
                pass


@app.route('/upload_redlight', methods=['POST'])
def upload_redlight():
    try:
        video_filename = request.form.get('video_path')
        line_coords = request.form.get('line_coords', '300,235,850,237')
        light_region = request.form.get('light_region', '100,50,200,150')
        coords = [int(x) for x in line_coords.split(',')]
        light_box = [int(x) for x in light_region.split(',')]
        video_path = os.path.join(app.config['TEMP_FOLDER'], video_filename)
        if not os.path.exists(video_path):
            return jsonify({'error': 'Video not found'}), 400
        print(f"Processing red light video: {video_filename}")
        print(f"Line coordinates: {coords}")
        print(f"Light region: {light_box}")
        challans = detect_red_light_violation(video_path, coords, light_box)
        violations_db.extend(challans)
        os.unlink(video_path)
        return jsonify({'success': True, 'challans': challans, 'total_challans': len(challans)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e), 'challans': []}), 500


@app.route('/violations')
def get_violations():
    return jsonify(violations_db)


@app.route('/image/<filename>')
def get_image(filename):
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], 'images', filename))


if __name__ == '__main__':
    app.run(debug=True, port=5000)