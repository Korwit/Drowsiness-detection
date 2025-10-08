"""import typer
import cv2
import time
from playsound import playsound
import supervision as sv
from ultralytics import YOLO
import mediapipe as mp
import numpy as np

# โหลดโมเดล YOLO
model = YOLO("drowsy.pt")
app = typer.Typer()

# Mediapipe สำหรับ landmark
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False)
MAR_THRESHOLD = 0.8  # ปรับตามการทดลอง

def calculate_mar(landmarks, image_shape):
    h, w = image_shape[:2]
    def xy(idx):
        return np.array([landmarks[idx].x * w, landmarks[idx].y * h])
    
    # ใช้ landmark รอบปากของ mediapipe
    # ตัวเลข landmark อ้างอิงจาก Mediapipe Face Mesh
    A = np.linalg.norm(xy(13) - xy(14))
    B = np.linalg.norm(xy(12) - xy(16))
    C = np.linalg.norm(xy(11) - xy(15))
    mar = (A + B) / (2.0 * C)
    return mar

def process_webcam(output_file="output.mp4"):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    drowsy_start_time = None
    alarm_played = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)

        labels = [
            f"{model.model.names[int(class_id)]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

        # ตรวจ class names
        class_names = [model.model.names[int(cid)] for cid in detections.class_id]

        yawning_detected = False
        eyes_closed_detected = False

        if "Drowsiness" in class_names:
            # เริ่มจับเวลา drowsy
            if drowsy_start_time is None:
                drowsy_start_time = time.time()
            elif time.time() - drowsy_start_time >= 3 and not alarm_played:
                # ใช้ Mediapipe ตรวจ MAR
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results_mesh = mp_face_mesh.process(rgb_frame)
                if results_mesh.multi_face_landmarks:
                    mar = calculate_mar(results_mesh.multi_face_landmarks[0].landmark, frame.shape)
                    if mar > MAR_THRESHOLD:
                        print("😮 Yawning detected!")
                        playsound("yawn.mp3")  # เสียงหาว
                        yawning_detected = True
                    else:
                        print("😴 Eyes closed detected!")
                        playsound("alarm.mp3")       # เสียงง่วง
                        eyes_closed_detected = True
                    alarm_played = True
        else:
            # รีเซ็ต
            drowsy_start_time = None
            alarm_played = False

        # วาด annotation
        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        # แสดงข้อความบนหน้าจอ
        if yawning_detected:
            cv2.putText(annotated_frame, "Yawning!", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        elif eyes_closed_detected:
            cv2.putText(annotated_frame, "Eyes Closed!", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        out.write(annotated_frame)
        cv2.imshow("Webcam", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

@app.command()
def webcam(output_file: str = "output.mp4"):
    typer.echo("Starting webcam processing...")
    process_webcam(output_file)

if __name__ == "__main__":
    app()"""



import typer
import cv2
import time
from playsound import playsound
import supervision as sv
from ultralytics import YOLO
import mediapipe as mp
import numpy as np

# โหลดโมเดล YOLO
model = YOLO("drowsy.pt")
app = typer.Typer()

# Mediapipe สำหรับ landmark
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False)
MAR_THRESHOLD = 0.78  # ปรับตามการทดลอง
EAR_THRESHOLD = 0.25  # ตาปิด
EAR_FRAMES = 3         # จำนวนเฟรมต่อเนื่องต้องตาปิดถึงถือว่า Eyes Closed
MAR_FRAMES = 3         # จำนวนเฟรมต่อเนื่องต้องหาวถึงถือว่า Yawning

# Counters
ear_counter = 0
mar_counter = 0

def calculate_mar(landmarks, image_shape):
    h, w = image_shape[:2]
    def xy(idx):
        return np.array([landmarks[idx].x * w, landmarks[idx].y * h])
    
    A = np.linalg.norm(xy(13) - xy(14))
    B = np.linalg.norm(xy(12) - xy(16))
    C = np.linalg.norm(xy(11) - xy(15))
    mar = (A + B) / (2.0 * C)
    return mar

def calculate_ear(landmarks, image_shape, eye='left'):
    h, w = image_shape[:2]
    def xy(idx):
        return np.array([landmarks[idx].x * w, landmarks[idx].y * h])

    if eye == 'left':
        # Mediapipe left eye landmarks
        p1, p2, p3, p4, p5, p6 = 33, 160, 158, 133, 153, 144
    else:
        # Mediapipe right eye landmarks
        p1, p2, p3, p4, p5, p6 = 362, 385, 387, 263, 373, 380

    # EAR formula
    A = np.linalg.norm(xy(p2) - xy(p6))
    B = np.linalg.norm(xy(p3) - xy(p5))
    C = np.linalg.norm(xy(p1) - xy(p4))
    ear = (A + B) / (2.0 * C)
    return ear

def process_webcam(output_file="output.mp4"):
    global ear_counter, mar_counter
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    drowsy_start_time = None
    alarm_played = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)

        labels = [
            f"{model.model.names[int(class_id)]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

        class_names = [model.model.names[int(cid)] for cid in detections.class_id]

        yawning_detected = False
        eyes_closed_detected = False

        if "Drowsiness" in class_names:
            if drowsy_start_time is None:
                drowsy_start_time = time.time()
            elif time.time() - drowsy_start_time >= 3 and not alarm_played:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results_mesh = mp_face_mesh.process(rgb_frame)
                if results_mesh.multi_face_landmarks:
                    landmarks = results_mesh.multi_face_landmarks[0].landmark
                    mar = calculate_mar(landmarks, frame.shape)
                    ear_left = calculate_ear(landmarks, frame.shape, 'left')
                    ear_right = calculate_ear(landmarks, frame.shape, 'right')
                    ear_avg = (ear_left + ear_right) / 2.0

                    # ตรวจหาว
                    if mar > MAR_THRESHOLD:
                        mar_counter += 1
                    else:
                        mar_counter = 0

                    # ตรวจตาปิด
                    if ear_avg < EAR_THRESHOLD:
                        ear_counter += 1
                    else:
                        ear_counter = 0

                    if mar_counter >= MAR_FRAMES:
                        print("😮 Yawning detected!")
                        playsound("yawn.mp3")
                        yawning_detected = True
                        alarm_played = True
                        ear_counter = 0  # รีเซ็ตตาปิดเพื่อไม่ให้ซ้ำ
                        mar_counter = 0
                    elif ear_counter >= EAR_FRAMES:
                        print("😴 Eyes closed detected!")
                        playsound("alarm.mp3")
                        eyes_closed_detected = True
                        alarm_played = True
                        ear_counter = 0

        else:
            drowsy_start_time = None
            alarm_played = False
            ear_counter = 0
            mar_counter = 0

        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        # แสดงข้อความบนหน้าจอ
        if yawning_detected:
            cv2.putText(annotated_frame, "Yawning!", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        elif eyes_closed_detected:
            cv2.putText(annotated_frame, "Eyes Closed!", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        out.write(annotated_frame)
        cv2.imshow("Webcam", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

@app.command()
def webcam(output_file: str = "output.mp4"):
    typer.echo("Starting webcam processing...")
    process_webcam(output_file)

if __name__ == "__main__":
    app()
