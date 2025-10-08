import argparse
from functools import partial
import tkinter as tk
import cv2
import os
import numpy as np
import face_recognition
import time
from playsound import playsound
import supervision as sv
from ultralytics import YOLO
import mediapipe as mp

# =================================
# Load YOLO Drowsiness Model
# =================================
model = YOLO("drowsy.pt")

# =================================
# Mediapipe Face Mesh
# =================================
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False)
MAR_THRESHOLD = 0.75
EAR_THRESHOLD = 0.25
MAR_FRAMES = 3
EAR_FRAMES = 3

# =================================
# Face Recognition Data
# =================================
DATA_DIR = "face_data"
os.makedirs(DATA_DIR, exist_ok=True)

# =================================
# Global State
# =================================
ear_counter = 0
mar_counter = 0
drowsy_start_time = None
yawn_start_time = None
alarm_played = False

# Face recognition tracking
face_locations_last = []
face_names_last = []
frame_count = 15
track_frame_interval = 15  # ตรวจใบหน้าใหม่ทุก 15 เฟรม

# =================================
# Utility Functions
# =================================
def calculate_mar(landmarks, image_shape):
    h, w = image_shape[:2]
    def xy(idx): return np.array([landmarks[idx].x * w, landmarks[idx].y * h])
    A = np.linalg.norm(xy(13)-xy(14))
    B = np.linalg.norm(xy(12)-xy(16))
    C = np.linalg.norm(xy(11)-xy(15))
    return (A+B)/(2.0*C)

def calculate_ear(landmarks, image_shape, eye='left'):
    h, w = image_shape[:2]
    def xy(idx): return np.array([landmarks[idx].x * w, landmarks[idx].y * h])
    if eye == 'left':
        p1,p2,p3,p4,p5,p6 = 33,160,158,133,153,144
    else:
        p1,p2,p3,p4,p5,p6 = 362,385,387,263,373,380
    A = np.linalg.norm(xy(p2)-xy(p6))
    B = np.linalg.norm(xy(p3)-xy(p5))
    C = np.linalg.norm(xy(p1)-xy(p4))
    return (A+B)/(2.0*C)

# =================================
# Face Registration
# =================================
def register_face(name):
    cap = cv2.VideoCapture(0)
    print(f"กำลังบันทึกใบหน้า: {name} (กด 's' เพื่อบันทึก, 'q' เพื่อออก)")
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
                print("❌ ไม่พบใบหน้า!")
                continue
            encodings = face_recognition.face_encodings(rgb, boxes)
            np.save(os.path.join(DATA_DIR, f"{name}.npy"), encodings[0])
            print(f"✅ บันทึกใบหน้า {name} แล้ว!")
            break
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# =================================
# Face Recognition with N-frame
# =================================
def recognize_face(frame):
    global face_locations_last, face_names_last, frame_count
    frame_count += 1

    # โหลด known faces
    known_faces = []
    known_names = []
    for file in os.listdir(DATA_DIR):
        if file.endswith(".npy"):
            known_faces.append(np.load(os.path.join(DATA_DIR, file)))
            known_names.append(file.replace(".npy", ""))

    # ตรวจใบหน้าใหม่ทุก track_frame_interval
    if frame_count % track_frame_interval == 0 or len(face_locations_last) == 0:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations_last = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations_last)
        face_names_last = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.45)
            name = "Unknown"
            if True in matches:
                name = known_names[matches.index(True)]
            face_names_last.append(name)

    # วาดใบหน้าและชื่อ
    for (top, right, bottom, left), name in zip(face_locations_last, face_names_last):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame, face_names_last

# =================================
# Webcam Drowsiness + Face Recognition
# =================================
def process_webcam(output_file="output.mp4"):
    global ear_counter, mar_counter, drowsy_start_time, yawn_start_time, alarm_played
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    box_annotator = sv.BoxAnnotator()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Face recognition
        frame, names_detected = recognize_face(frame)

        # ตรวจเฉพาะคนลงทะเบียนแล้ว
        if "Unknown" not in names_detected and len(names_detected) > 0:
            results = model(frame)[0]
            detections = sv.Detections.from_ultralytics(results)
            class_names = [model.model.names[int(c)] for c in detections.class_id]

            if "Drowsiness" in class_names:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_results = mp_face_mesh.process(rgb_frame)
                if mp_results.multi_face_landmarks:
                    landmarks = mp_results.multi_face_landmarks[0].landmark
                    mar = calculate_mar(landmarks, frame.shape)
                    ear_left = calculate_ear(landmarks, frame.shape, 'left')
                    ear_right = calculate_ear(landmarks, frame.shape, 'right')
                    ear_avg = (ear_left+ear_right)/2.0

                    # MAR / EAR counters
                    global mar_counter, ear_counter
                    if mar > MAR_THRESHOLD:
                        mar_counter += 1
                    else:
                        mar_counter = 0

                    if ear_avg < EAR_THRESHOLD:
                        ear_counter += 1
                    else:
                        ear_counter = 0

                    # Yawning detected N frames
                    global yawn_start_time, drowsy_start_time
                    if mar_counter >= MAR_FRAMES and not alarm_played:
                        print("😮 Yawning detected!")
                        playsound("yawn.mp3")
                        alarm_played = True
                        mar_counter = 0
                        yawn_start_time = None

                    # Eyes Closed detected N frames
                    if ear_counter >= EAR_FRAMES and not alarm_played:
                        print("😴 Eyes closed detected!")
                        playsound("alarm.mp3")
                        alarm_played = True
                        ear_counter = 0
                        drowsy_start_time = None
        else:
            # คน Unknown ไม่ตรวจ
            ear_counter = 0
            mar_counter = 0
            drowsy_start_time = None
            yawn_start_time = None
            alarm_played = False

        annotated = box_annotator.annotate(
            scene=frame.copy(),
            detections=sv.Detections.from_ultralytics(model(frame)[0])
        )
        cv2.imshow("Webcam", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# =================================
# Tkinter GUI
# =================================
def start_register(name_entry):
    name = name_entry.get()
    if name.strip():
        register_face(name)

def open_register_ui():
    root = tk.Tk()
    root.title("Driver Monitoring - Register Face")

    tk.Label(root, text="Enter Name:").pack()
    name_entry = tk.Entry(root)
    name_entry.pack()

    tk.Button(root, text="Register Face", command=partial(start_register, name_entry)).pack()
    tk.Button(root, text="Close", command=root.destroy).pack()

    root.mainloop()

# =================================
# Main CLI
# =================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["webcam", "register"], help="Mode: 'webcam' to start camera, 'register' to register face")
    args = parser.parse_args()

    if args.mode == "webcam":
        print("Starting webcam with drowsiness + face recognition...")
        process_webcam("output.mp4")
    elif args.mode == "register":
        open_register_ui()
