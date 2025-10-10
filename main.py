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
import time
import csv
from datetime import datetime
import threading
import tkinter.simpledialog as simpledialog
import subprocess
import platform

# =================================
# Load YOLO Drowsiness Model
# =================================
model = YOLO("model.pt")

# =================================
# Mediapipe Face Mesh
# =================================
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False)
MAR_THRESHOLD = 0.78
EAR_THRESHOLD = 0.25
MAR_FRAMES = 3
EAR_FRAMES = 3

# =================================
# Face Recognition Data
# =================================
FACE_DIR = "face_data"
USER_DIR = "user_data"

os.makedirs(FACE_DIR, exist_ok=True)
os.makedirs(USER_DIR, exist_ok=True)


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
last_yawn_time = 0  
alarm_running = False

def play_sound(file_path):
    global alarm_running
    if not alarm_running:
        def target():
            global alarm_running
            alarm_running = True
            playsound(file_path)
            alarm_running = False
        threading.Thread(target=target, daemon=True).start()


def log_drowsiness_event(user_name, event_type):
    if not user_name:
        user_name = "Unknown"
    file_path = os.path.join(USER_DIR, f"{user_name}.csv")  # เปลี่ยนโฟลเดอร์
    now = datetime.now()

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
            count = len(rows) if len(rows) > 1 else 1
    else:
        count = 1

    row = [count, now.strftime("%d/%m/%Y"), now.strftime("%H:%M:%S"), event_type]

    file_exists = os.path.exists(file_path)
    with open(file_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["number", "Date", "Time", "Event"])
        writer.writerow(row)




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
    file_path = os.path.join(FACE_DIR, f"{name}.npy")
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
            for file in os.listdir(FACE_DIR):
                if file.endswith(".npy"):
                    known_faces.append(np.load(os.path.join(FACE_DIR, file)))
                    known_names.append(file.replace(".npy", ""))

            duplicate_found = False
            for existing_face, existing_name in zip(known_faces, known_names):
                match = face_recognition.compare_faces([existing_face], encodings[0], tolerance=0.45)
                if match[0]:
                    duplicate_found = True
                    messagebox.showwarning("พบใบหน้าซ้ำ", f"ใบหน้านี้เคยลงทะเบียนเป็น '{existing_name}' แล้ว!")
                    print(f"ใบหน้านี้ตรงกับ {existing_name} ที่มีอยู่แล้ว")
                    
                    # นำ focus กลับมาที่ OpenCV
                    cv2.namedWindow("Register Face", cv2.WINDOW_NORMAL)
                    cv2.setWindowProperty("Register Face", cv2.WND_PROP_TOPMOST, 1)
                    cv2.setWindowProperty("Register Face", cv2.WND_PROP_TOPMOST, 0)
                    
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
    for file in os.listdir(FACE_DIR):
        if file.endswith(".npy"):
            known_faces.append(np.load(os.path.join(FACE_DIR, file)))
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


def process_webcam():
    global ear_counter, mar_counter, drowsy_start_time, yawn_start_time
    global frame_count, face_recognition_locked, current_user_known, known_user_names,last_yawn_time

    # แยก flag สำหรับ alarm แต่ละประเภท
    alarm_played_ear = False
    alarm_played_mar = False

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    frame_count = 0
    face_recognition_locked = False
    current_user_known = False
    known_user_names = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        yawning_detected = False
        eyes_closed_detected = False

        # =========================
        # Face recognition ทุก ๆ track_frame_interval เฟรม
        # =========================
        if not face_recognition_locked:
            if frame_count == 1 or frame_count % track_frame_interval == 0:
                frame, names_detected = recognize_face_once(frame)
                if "Unknown" not in names_detected and len(names_detected) > 0:
                    known_user_names = names_detected
                    play_sound("sound/hi.mp3")
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

                    current_time = time.time()

                   # =========================
                    # MAR / EAR counters
                    # =========================
                    # ตรวจหาว
                    if mar > MAR_THRESHOLD:
                        mar_counter += 1
                        if yawn_start_time is None:
                            yawn_start_time = current_time
                        elif mar_counter >= MAR_FRAMES and current_time - yawn_start_time >= 0.5 and not alarm_played_mar:
                            print("Yawning detected!")
                            play_sound("sound/yawn.mp3")
                            last_yawn_time = current_time
                            yawning_detected = True
                            alarm_played_mar = True
                            if known_user_names:
                                log_drowsiness_event(known_user_names[0], "Yawning")
                    else:
                        mar_counter = 0
                        yawn_start_time = None
                        alarm_played_mar = False

                    # -------------------------
                    # ตรวจตาปิด / ง่วงนอน
                    # -------------------------
                    if ear_avg < EAR_THRESHOLD:
                        # ตรวจว่าเสียง yawn เล่นจบไปแล้ว 2 วินาทีขึ้นไป และ alarm ไม่ซ้อน
                        if current_time - last_yawn_time >= 2.5 and not alarm_running:
                            if drowsy_start_time is None:
                                drowsy_start_time = current_time

                            if current_time - drowsy_start_time >= 2.0:
                                print("Eyes closed detected! Alarm!")
                                play_sound("sound/alarm.mp3")
                                if known_user_names:
                                    log_drowsiness_event(known_user_names[0], "Eyes Closed")
                                drowsy_start_time = current_time
                    else:
                        drowsy_start_time = None


        # =========================
        # Annotate frame (Box + Label)
        # =========================
        results_for_annotate = model(frame)[0]
        detections_for_annotate = sv.Detections.from_ultralytics(results_for_annotate)
        labels = [f"{model.model.names[int(cid)]} {conf:.2f}" 
                  for cid, conf in zip(detections_for_annotate.class_id, detections_for_annotate.confidence)]

        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections_for_annotate)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections_for_annotate, labels=labels)

        # ข้อความสีแดงบนหน้าจอ
        if yawning_detected:
            cv2.putText(annotated_frame, "Yawning!", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
        elif eyes_closed_detected:
            cv2.putText(annotated_frame, "Eyes Closed!", (30,50), cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),3)

        cv2.imshow("Webcam", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# =================================
# Tkinter GUI with user management
# =================================
def start_register(name_entry, user_listbox):
    name = name_entry.get().strip()
    if not name:
        messagebox.showwarning("ข้อผิดพลาด", "กรุณากรอกชื่อก่อน!")
        return
    # ล้าง Entry ทันที
    name_entry.delete(0, tk.END)

    register_face(name)
    refresh_user_list(user_listbox)

def refresh_user_list(listbox):
    listbox.delete(0, tk.END)
    for file in os.listdir(FACE_DIR):
        if file.endswith(".npy"):
            name = file.replace(".npy", "")
            listbox.insert(tk.END, name)

def delete_user(listbox):
    selected = listbox.curselection()
    if not selected:
        messagebox.showwarning("ข้อผิดพลาด", "กรุณาเลือกผู้ใช้ก่อน")
        return
    name = listbox.get(selected[0])
    confirm = messagebox.askyesno("ยืนยันการลบ", f"คุณต้องการลบผู้ใช้ '{name}' ใช่หรือไม่?")
    if confirm:
        npy_file = os.path.join(FACE_DIR, f"{name}.npy")
        csv_file = os.path.join(USER_DIR, f"{name}.csv")  # เปลี่ยนโฟลเดอร์ CSV
        for f in [npy_file, csv_file]:
            if os.path.exists(f):
                os.remove(f)
        refresh_user_list(listbox)
        messagebox.showinfo("สำเร็จ", f"ผู้ใช้ '{name}' ถูกลบแล้ว")

def rename_user(listbox):
    selected = listbox.curselection()
    if not selected:
        messagebox.showwarning("ข้อผิดพลาด", "กรุณาเลือกผู้ใช้ก่อน")
        return
    old_name = listbox.get(selected[0])
    new_name = simpledialog.askstring("เปลี่ยนชื่อ", f"เปลี่ยนชื่อ '{old_name}' เป็น:")
    if not new_name:
        return

    if os.path.exists(os.path.join(FACE_DIR, f"{new_name}.npy")):
        messagebox.showwarning("ชื่อซ้ำ", f"ชื่อ '{new_name}' มีอยู่แล้ว")
        return

    old_npy = os.path.join(FACE_DIR, f"{old_name}.npy")
    old_csv = os.path.join(USER_DIR, f"{old_name}.csv")
    new_npy = os.path.join(FACE_DIR, f"{new_name}.npy")
    new_csv = os.path.join(USER_DIR, f"{new_name}.csv")
    if os.path.exists(old_npy):
        os.rename(old_npy, new_npy)
    if os.path.exists(old_csv):
        os.rename(old_csv, new_csv)

    refresh_user_list(listbox)
    messagebox.showinfo("สำเร็จ", f"เปลี่ยนชื่อ '{old_name}' เป็น '{new_name}' เรียบร้อยแล้ว")


def open_csv_window():
    csv_window = tk.Toplevel()
    csv_window.title("Open CSV Files")
    csv_window.geometry("400x300")

    tk.Label(csv_window, text="Available CSV files:").pack(pady=5)

    csv_listbox = tk.Listbox(csv_window, width=20, height=10)
    csv_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10,0))

    scrollbar = tk.Scrollbar(csv_window)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    csv_listbox.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=csv_listbox.yview)

    # โหลด CSV files จาก USER_DIR
    csv_files = [f for f in os.listdir(USER_DIR) if f.endswith(".csv")]
    for f in csv_files:
        csv_listbox.insert(tk.END, f)

    def open_selected_csv():
        selected = csv_listbox.curselection()
        if not selected:
            tk.messagebox.showwarning("คำเตือน", "กรุณาเลือกไฟล์ CSV ก่อนเปิดใน Excel!")
            return

        file_path = os.path.join(USER_DIR, csv_listbox.get(selected[0]))
        system_name = platform.system()

        try:
            if system_name == "Windows":
                os.startfile(file_path)
            elif system_name == "Darwin":
                subprocess.call(["open", file_path])
            else:
                subprocess.call(["xdg-open", file_path])
        except Exception as e:
            tk.messagebox.showerror("ข้อผิดพลาด", f"ไม่สามารถเปิดไฟล์ได้:\n{e}")

    tk.Button(csv_window, text="Open in Excel", command=open_selected_csv).pack(pady=5)


def open_register_ui():
    root = tk.Tk()
    root.title("Driver Monitoring")

    # ตั้งขนาดหน้าต่างเริ่มต้น
    root.geometry("350x400")  # กว้าง 350 px, สูง 400 px

    # ---------- Entry + Register ----------
    tk.Label(root, text="Enter Name:").pack(pady=5)
    name_entry = tk.Entry(root)
    name_entry.pack(pady=5)

    tk.Button(root, text="Register Face", command=lambda: start_register(name_entry, user_listbox)).pack(pady=5)

    # ---------- Frame สำหรับ Listbox + Scrollbar ----------
    list_frame = tk.Frame(root)
    list_frame.pack(pady=5)

    user_listbox = tk.Listbox(list_frame, width=30, height=10)
    user_listbox.pack(side=tk.LEFT, fill=tk.BOTH)

    scrollbar = tk.Scrollbar(list_frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # เชื่อม Listbox กับ Scrollbar
    user_listbox.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=user_listbox.yview)

    refresh_user_list(user_listbox)

    # ---------- ปุ่ม Delete, Rename, Close ----------
    tk.Button(root, text="Delete User", command=lambda: delete_user(user_listbox)).pack(pady=2)
    tk.Button(root, text="Rename User", command=lambda: rename_user(user_listbox)).pack(pady=2)
    tk.Button(root, text="Open CSV", command=open_csv_window).pack(pady=5)

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
        process_webcam()
    elif args.mode == "register":
        open_register_ui()
