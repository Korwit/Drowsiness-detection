import typer
import cv2
import time
from playsound import playsound
import supervision as sv
from ultralytics import YOLO

model = YOLO("drowsy.pt")
app = typer.Typer()

def process_webcam(output_file="output.mp4"):
    cap = cv2.VideoCapture(0)  #  เปิดเว็บแคม

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

    drowsy_start_time = None     # ⏱ เวลาที่เริ่มตรวจพบ drowsiness
    alarm_played = False         # ป้องกันเล่นเสียงซ้ำ

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

        # ✅ ตรวจว่ามี class "drowsiness" หรือไม่
        class_names = [model.model.names[int(cid)] for cid in detections.class_id]
        if "Drowsiness" in class_names:
            if drowsy_start_time is None:
                drowsy_start_time = time.time()  # เริ่มจับเวลา
            elif time.time() - drowsy_start_time >= 3 and not alarm_played:
                print("⚠️ Drowsiness detected for 3 seconds! Playing alarm...")
                playsound("alarm.mp3")  # เล่นเสียงเตือน
                alarm_played = True
        else:
            # ถ้าไม่พบ drowsiness ให้รีเซ็ตเวลา
            drowsy_start_time = None
            alarm_played = False

        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

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





