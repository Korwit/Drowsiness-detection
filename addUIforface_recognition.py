import argparse
from functools import partial
import tkinter as tk
from tkinter import messagebox
import cv2
import os
import numpy as np
import face_recognition
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

frame_count = 0
track_frame_interval = 10  # ตรวจใบหน้าใหม่ทุก 10 เฟรม

# Face recognition flags
face_recognition_locked = False  # True ถ้าเจอ known user แล้ว
current_user_known = False
known_user_names = []

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
# Face Registration (ป้องกันชื่อซ้ำ)
# =================================
def is_face_straight(landmarks, image_shape, threshold=0.15):
    """ตรวจว่าหน้าตรงหรือไม่"""
    h, w = image_shape[:2]

    def xy(idx):
        return np.array([landmarks[idx].x * w, landmarks[idx].y * h])

    nose_tip = xy(1)
    left_eye = xy(33)
    right_eye = xy(263)

    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    face_ratio = abs(nose_tip[0] - eye_center_x) / np.linalg.norm(right_eye - left_eye)
    return face_ratio < threshold  # True = หน้าตรง

def register_face(name):
    file_path = os.path.join(DATA_DIR, f"{name}.npy")
    if os.path.exists(file_path):
        messagebox.showwarning("ชื่อซ้ำ", f"ชื่อ '{name}' มีอยู่แล้ว!\nกรุณาใช้ชื่ออื่น")
        print(f"ชื่อ '{name}' มีอยู่แล้ว!")
        return

    cap = cv2.VideoCapture(0)
    print(f"กำลังบันทึกใบหน้า: {name} (กด 'S' เพื่อบันทึก, 'Q' เพื่อออก)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame, "Press 'S' to Save  |  'Q' to Quit",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_results = mp_face_mesh.process(rgb)

        if mp_results.multi_face_landmarks:
            landmarks = mp_results.multi_face_landmarks[0].landmark

            # ----- ตรวจห้ามยิ้ม/อ้าปาก -----
            mar = calculate_mar(landmarks, frame.shape)
            if mar > MAR_THRESHOLD:
                cv2.putText(frame, "Please close your mouth / no big smile",
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow("Register Face", frame)
                cv2.waitKey(1)
                continue

            # ----- ตรวจหน้าตรง -----
            if not is_face_straight(landmarks, frame.shape, threshold=0.15):
                cv2.putText(frame, "Please face straight forward",
                            (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow("Register Face", frame)
                cv2.waitKey(1)
                continue

            # วาดกรอบรอบหน้า
            h, w, _ = frame.shape
            x_min = min([lm.x for lm in landmarks]) * w
            x_max = max([lm.x for lm in landmarks]) * w
            y_min = min([lm.y for lm in landmarks]) * h
            y_max = max([lm.y for lm in landmarks]) * h
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

        cv2.imshow("Register Face", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            # ตรวจใบหน้าซ้ำ
            boxes = face_recognition.face_locations(rgb)
            if len(boxes) == 0:
                print("❌ ไม่พบใบหน้า!")
                continue

            encodings = face_recognition.face_encodings(rgb, boxes)

            known_faces = []
            known_names = []
            for file in os.listdir(DATA_DIR):
                if file.endswith(".npy"):
                    known_faces.append(np.load(os.path.join(DATA_DIR, file)))
                    known_names.append(file.replace(".npy", ""))

            duplicate_found = False
            for existing_face, existing_name in zip(known_faces, known_names):
                match = face_recognition.compare_faces([existing_face], encodings[0], tolerance=0.45)
                if match[0]:
                    duplicate_found = True
                    messagebox.showwarning("พบใบหน้าซ้ำ", f"ใบหน้านี้เคยลงทะเบียนเป็น '{existing_name}' แล้ว!")
                    print(f"ใบหน้านี้ตรงกับ {existing_name} ที่มีอยู่แล้ว")
                    break

            if duplicate_found:
                continue

            # บันทึก
            np.save(file_path, encodings[0])
            print(f"✅ บันทึกใบหน้า {name} แล้ว!")
            messagebox.showinfo("สำเร็จ", f"บันทึกใบหน้า {name} เรียบร้อยแล้ว!")
            break

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()




# =================================
# Face Recognition (เรียกครั้งเดียว)
# =================================
def recognize_face_once(frame):
    known_faces = []
    known_names = []
    for file in os.listdir(DATA_DIR):
        if file.endswith(".npy"):
            known_faces.append(np.load(os.path.join(DATA_DIR, file)))
            known_names.append(file.replace(".npy", ""))

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    face_names = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.45)
        name = "Unknown"
        if True in matches:
            name = known_names[matches.index(True)]
        face_names.append(name)

    # วาดใบหน้าและชื่อ
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame, face_names

# =================================
# Webcam Drowsiness + Face Recognition
# =================================
def process_webcam(output_file="output.mp4"):
    global ear_counter, mar_counter, drowsy_start_time, yawn_start_time
    global frame_count, face_recognition_locked, current_user_known, known_user_names

    # แยก flag สำหรับ alarm แต่ละประเภท
    alarm_played_ear = False
    alarm_played_mar = False

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    box_annotator = sv.BoxAnnotator()
    frame_count = 0
    face_recognition_locked = False
    current_user_known = False
    known_user_names = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # =========================
        # Face recognition ทุก ๆ track_frame_interval เฟรม
        # =========================
        if not face_recognition_locked:
            if frame_count == 1 or frame_count % track_frame_interval == 0:
                frame, names_detected = recognize_face_once(frame)
                if "Unknown" not in names_detected and len(names_detected) > 0:
                    known_user_names = names_detected
                    face_recognition_locked = True
                    current_user_known = True
                    print(f"✅ Registered user detected: {names_detected}")
                else:
                    known_user_names = []
                    current_user_known = False

        # =========================
        # Drowsiness detection เฉพาะ known user
        # =========================
        if current_user_known:
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
                    ear_avg = (ear_left + ear_right) / 2.0

                    # =========================
                    # MAR / EAR counters
                    # =========================
                    if mar > MAR_THRESHOLD:
                        mar_counter += 1
                    else:
                        mar_counter = 0
                        alarm_played_mar = False  # reset flag เมื่อเหตุการณ์จบ

                    if ear_avg < EAR_THRESHOLD:
                        ear_counter += 1
                    else:
                        ear_counter = 0
                        alarm_played_ear = False  # reset flag เมื่อเหตุการณ์จบ

                    # =========================
                    # Alarm
                    # =========================
                    if mar_counter >= MAR_FRAMES and not alarm_played_mar:
                        print("😮 Yawning detected!")
                        playsound("yawn.mp3")
                        alarm_played_mar = True
                        yawn_start_time = None

                    if ear_counter >= EAR_FRAMES and not alarm_played_ear:
                        print("😴 Eyes closed detected!")
                        playsound("alarm.mp3")
                        alarm_played_ear = True
                        drowsy_start_time = None

        # =========================
        # Annotate frame
        # =========================
        results_for_annotate = model(frame)[0]
        annotated = box_annotator.annotate(
            scene=frame.copy(),
            detections=sv.Detections.from_ultralytics(results_for_annotate)
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
    name = name_entry.get().strip()
    if not name:
        messagebox.showwarning("ข้อผิดพลาด", "กรุณากรอกชื่อก่อน!")
        return
    register_face(name)

def open_register_ui():
    root = tk.Tk()
    root.title("Driver Monitoring - Register Face")

    tk.Label(root, text="Enter Name:").pack(pady=5)
    name_entry = tk.Entry(root)
    name_entry.pack(pady=5)

    tk.Button(root, text="Register Face", command=partial(start_register, name_entry)).pack(pady=5)
    tk.Button(root, text="Close", command=root.destroy).pack(pady=5)

    root.mainloop()

# =================================
# Main CLI
# =================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["webcam", "register"],
                        help="Mode: 'webcam' to start camera, 'register' to register face")
    args = parser.parse_args()

    if args.mode == "webcam":
        print("Starting webcam with drowsiness + face recognition...")
        process_webcam("output.mp4")
    elif args.mode == "register":
        open_register_ui()
