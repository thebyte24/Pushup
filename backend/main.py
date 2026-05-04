"""
FastAPI backend for AI Pushup Counter.
Accepts base64-encoded frames via WebSocket and streams back rep count + form data.
"""

import cv2
import numpy as np
import base64
import time
import os
import csv
import urllib.request
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

MODEL_PATH = os.environ.get("MODEL_PATH", "pose_landmarker_full.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/"
    "pose_landmarker_full.task"
)

def ensure_model():
    if not os.path.isfile(MODEL_PATH):
        print("[INFO] Downloading pose model (~30 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[INFO] Model ready.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ELBOW_DOWN     = 90
ELBOW_UP       = 160
HIP_SAG_PX     = 50
HEAD_DROP_MARGIN = 30
WRIST_ALIGN_TOL  = 80

R_EAR=8; R_SHOULDER=12; R_ELBOW=14; R_WRIST=16; R_HIP=24; R_ANKLE=28
L_EAR=7; L_SHOULDER=11; L_ELBOW=13; L_WRIST=15; L_HIP=23; L_ANKLE=27

HISTORY_FILE = os.environ.get("HISTORY_FILE", "workout_history.csv")
CSV_HEADERS  = ["Date", "Total_Reps", "Duration_Seconds"]

# ---------------------------------------------------------------------------
# Math helpers (same as pushup_counter.py)
# ---------------------------------------------------------------------------

def calculate_angle(a, b, c):
    a = np.array([a[0], a[1], 0.0])
    b = np.array([b[0], b[1], 0.0])
    c = np.array([c[0], c[1], 0.0])
    ba = a - b; bc = c - b
    cross = np.cross(ba, bc)[2]
    dot   = np.dot(ba, bc)
    return abs(np.degrees(np.arctan2(cross, dot)))

def point_to_line_distance(p, line_a, line_b):
    p = np.array(p, dtype=float)
    line_a = np.array(line_a, dtype=float)
    line_b = np.array(line_b, dtype=float)
    d = line_b - line_a
    norm = np.linalg.norm(d)
    if norm == 0:
        return float(np.linalg.norm(p - line_a))
    return float(abs(np.cross(d, line_a - p)) / norm)

def check_hip_alignment(shoulder, hip, ankle):
    dist = point_to_line_distance(hip, shoulder, ankle)
    return dist, dist < HIP_SAG_PX

def check_head_position(ear, shoulder, hip):
    mid_y = (shoulder[1] + hip[1]) / 2.0
    diff  = mid_y - ear[1]
    return diff, diff > -HEAD_DROP_MARGIN

def check_wrist_stack(shoulder, wrist):
    diff = abs(shoulder[0] - wrist[0])
    return diff, diff < WRIST_ALIGN_TOL

def get_side_landmarks(lms, w, h):
    def vis(idx):
        return lms[idx].visibility if lms[idx].visibility is not None else 0.0
    right_score = vis(R_SHOULDER) + vis(R_ELBOW) + vis(R_HIP) + vis(R_ANKLE)
    left_score  = vis(L_SHOULDER) + vis(L_ELBOW) + vis(L_HIP) + vis(L_ANKLE)
    use_right   = right_score >= left_score
    ids = (R_EAR, R_SHOULDER, R_ELBOW, R_WRIST, R_HIP, R_ANKLE) if use_right else \
          (L_EAR, L_SHOULDER, L_ELBOW, L_WRIST, L_HIP, L_ANKLE)
    def c(idx):
        lm = lms[idx]
        return [lm.x * w, lm.y * h]
    return {k: c(ids[i]) for i, k in enumerate(["ear","shoulder","elbow","wrist","hip","ankle"])}

def save_session(total_reps, duration_seconds):
    file_exists = os.path.isfile(HISTORY_FILE)
    with open(HISTORY_FILE, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Total_Reps": total_reps,
            "Duration_Seconds": round(duration_seconds, 2),
        })

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# REST: workout history
# ---------------------------------------------------------------------------

@app.get("/history")
def get_history():
    if not os.path.isfile(HISTORY_FILE):
        return []
    rows = []
    with open(HISTORY_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


@app.delete("/history/{date}")
def delete_session(date: str):
    if not os.path.isfile(HISTORY_FILE):
        return {"ok": False}
    rows = []
    with open(HISTORY_FILE, newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r["Date"] != date]
    with open(HISTORY_FILE, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(rows)
    return {"ok": True}

# ---------------------------------------------------------------------------
# WebSocket: live pushup detection
# ---------------------------------------------------------------------------

@app.websocket("/ws/pushup")
async def pushup_ws(websocket: WebSocket):
    await websocket.accept()
    ensure_model()

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    rep_count      = 0
    movement_state = None
    form_failed    = False
    form_fail_reason = ""
    start_time     = time.time()
    feedback       = ""

    try:
        while True:
            # Receive base64 frame from React
            data = await websocket.receive_text()
            img_bytes = base64.b64decode(data)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            h, w = frame.shape[:2]
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms  = int((time.time() - start_time) * 1000)
            result = landmarker.detect_for_video(mp_img, ts_ms)

            response = {
                "reps": rep_count,
                "state": movement_state,
                "feedback": feedback,
                "pose_detected": False,
                "checks": {},
            }

            if result.pose_landmarks:
                lms  = result.pose_landmarks[0]
                pts  = get_side_landmarks(lms, w, h)

                shoulder = pts["shoulder"]; elbow = pts["elbow"]
                wrist    = pts["wrist"];    hip   = pts["hip"]
                ankle    = pts["ankle"];    ear   = pts["ear"]

                elbow_angle          = calculate_angle(shoulder, elbow, wrist)
                hip_dist,  hip_ok    = check_hip_alignment(shoulder, hip, ankle)
                head_diff, head_ok   = check_head_position(ear, shoulder, hip)
                stack_diff, stack_ok = check_wrist_stack(shoulder, wrist)

                # State machine
                if elbow_angle <= ELBOW_DOWN and movement_state != "DOWN":
                    movement_state = "DOWN"
                    form_failed = False
                    form_fail_reason = ""

                if movement_state == "DOWN":
                    if not hip_ok and not form_failed:
                        form_failed = True
                        form_fail_reason = "Hip sagging — keep body straight"
                    elif not head_ok and not form_failed:
                        form_failed = True
                        form_fail_reason = "Head dropping — keep neck neutral"

                if elbow_angle >= ELBOW_UP and movement_state == "DOWN":
                    movement_state = "UP"
                    if not stack_ok and not form_failed:
                        form_failed = True
                        form_fail_reason = "Wrist not under shoulder"
                    if not form_failed:
                        rep_count += 1
                        feedback = f"Good rep! #{rep_count}"
                    else:
                        feedback = form_fail_reason

                # Landmarks for frontend skeleton overlay
                landmarks = [{"x": lm.x, "y": lm.y} for lm in lms]

                response = {
                    "reps": rep_count,
                    "state": movement_state,
                    "feedback": feedback,
                    "pose_detected": True,
                    "checks": {
                        "elbow_angle": round(elbow_angle, 1),
                        "hip_ok": hip_ok,
                        "hip_dist": round(hip_dist, 1),
                        "head_ok": head_ok,
                        "stack_ok": stack_ok,
                        "stack_diff": round(stack_diff, 1),
                    },
                    "landmarks": landmarks,
                }

            await websocket.send_json(response)

    except WebSocketDisconnect:
        elapsed = time.time() - start_time
        save_session(rep_count, elapsed)
        landmarker.close()
        print(f"[INFO] Session ended. Reps: {rep_count}")
