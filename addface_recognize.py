import typer
import cv2
import time
import os
import numpy as np
import face_recognition
from playsound import playsound
import supervision as sv
from ultralytics import YOLO
import mediapipe as mp

model = YOLO("drowsy.pt")
app = typer.Typer()

# === Mediapipe face mesh ===
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False)
MAR_THRESHOLD = 0.75
EAR_THRESHOLD = 0.25
MAR_FRAMES = 3
EAR_FRAMES = 3

# === Face Recognition Data Path ===
DATA_DIR = "face_data"
os.makedirs(DATA_DIR, exist_ok=True)

# === Global state ===
ear_counter = 0
mar_counter = 0
drowsy_start_time = None
yawn_start_time = None
alarm_played = False

# ==========================================
# Utility Functions
# ==========================================
def calculate_mar(landmarks, image_shape):
    h, w = image_shape[:2]
    def xy(idx): return np.array([landmarks[idx].x * w, landmarks[idx].y * h])
    A = np.linalg.norm(xy(13) - xy(14))
    B = np.linalg.norm(xy(12) - xy(16))
    C = np.linalg.norm(xy(11) - xy(15))
    return (A + B) / (2.0 * C)

def calculate_ear(landmarks, image_shape, eye='left'):
    h, w = image_shape[:2]
    def xy(idx): return np.array([landmarks[idx].x * w, landmarks[idx].y * h])
    if eye == 'left':
        p1, p2, p3, p4, p5, p6 = 33,160,158,133,153,144
    else:
        p1, p2, p3, p4, p5, p6 = 362,385,387,263,373,380
    A = np.linalg.norm(xy(p2)-xy(p6))
    B = np.linalg.norm(xy(p3)-xy(p5))
    C = np.linalg.norm(xy(p1)-xy(p4))
    return (A+B)/(2.0*C)

# ==========================================
# Face Recognition Functions
# ==========================================
def register_face(name: str):
    """📸 ลงทะเบียนใบหน้าใหม่"""
    cap = cv2.VideoCapture(0)
    typer.echo(f"กำลังบันทึกใบหน้า: {name} (กด 's' เพื่อบันทึก, 'q' เพื่อออก)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Register Face", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb)
            if len(boxes) == 0:
                typer.echo("❌ ไม่พบใบหน้า!")
                continue
            encodings = face_recognition.face_encodings(rgb, boxes)
            np.save(os.path.join(DATA_DIR, f"{name}.npy"), encodings[0])
            typer.echo(f"✅ บันทึกใบหน้า {name} แล้ว!")
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def recognize_face(frame):
    """🔍 ตรวจสอบว่าใบหน้าตรงกับใคร"""
    known_faces = []
    known_names = []
    for file in os.listdir(DATA_DIR):
        if file.endswith(".npy"):
            known_faces.append(np.load(os.path.join(DATA_DIR, file)))
            known_names.append(file.replace(".npy", ""))

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.45)
        name = "Unknown"
        if True in matches:
            name = known_names[matches.index(True)]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    return frame

# ==========================================
# Main Drowsiness + Face Recognition
# ==========================================
def process_webcam(output_file="output.mp4"):
    global ear_counter, mar_counter, drowsy_start_time, yawn_start_time, alarm_played
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # === Face recognition overlay ===
        frame = recognize_face(frame)

        # === YOLO drowsiness detection ===
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        labels = [f"{model.model.names[int(c)]} {conf:.2f}" for c, conf in zip(detections.class_id, detections.confidence)]
        class_names = [model.model.names[int(c)] for c in detections.class_id]

        yawning_detected = False
        eyes_closed_detected = False

        if "Drowsiness" in class_names:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_mesh = mp_face_mesh.process(rgb_frame)
            if results_mesh.multi_face_landmarks:
                landmarks = results_mesh.multi_face_landmarks[0].landmark
                mar = calculate_mar(landmarks, frame.shape)
                ear_left = calculate_ear(landmarks, frame.shape, 'left')
                ear_right = calculate_ear(landmarks, frame.shape, 'right')
                ear_avg = (ear_left + ear_right) / 2.0

                if mar > MAR_THRESHOLD:
                    mar_counter += 1
                else:
                    mar_counter = 0
                if ear_avg < EAR_THRESHOLD:
                    ear_counter += 1
                else:
                    ear_counter = 0

                current_time = time.time()
                if mar_counter >= MAR_FRAMES:
                    if yawn_start_time is None:
                        yawn_start_time = current_time
                    elif current_time - yawn_start_time >= 1.0 and not alarm_played:
                        print("😮 Yawning detected!")
                        playsound("yawn.mp3")
                        yawning_detected = True
                        alarm_played = True
                        mar_counter = 0
                        yawn_start_time = None
                else:
                    yawn_start_time = None

                if ear_counter >= EAR_FRAMES:
                    if drowsy_start_time is None:
                        drowsy_start_time = current_time
                    elif current_time - drowsy_start_time >= 2.5 and not alarm_played:
                        print("😴 Eyes closed detected!")
                        playsound("alarm.mp3")
                        eyes_closed_detected = True
                        alarm_played = True
                        ear_counter = 0
                        drowsy_start_time = None
                else:
                    drowsy_start_time = None

        else:
            ear_counter = 0
            mar_counter = 0
            drowsy_start_time = None
            yawn_start_time = None
            alarm_played = False

        annotated = box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

        if yawning_detected:
            cv2.putText(annotated, "Yawning!", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255),3)
        elif eyes_closed_detected:
            cv2.putText(annotated, "Eyes Closed!", (30,50), cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),3)

        cv2.imshow("Webcam", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ==========================================
# Typer Commands
# ==========================================
@app.command()
def webcam(output_file: str = "output.mp4"):
    typer.echo("Starting webcam with drowsiness + face recognition...")
    process_webcam(output_file)

@app.command()
def register(name: str):
    register_face(name)

if __name__ == "__main__":
    app()
