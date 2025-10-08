import cv2

# เลือกกล้อง C922 ตาม index จาก list
camera_index = 0
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print("ไม่สามารถเปิดกล้อง C922 ได้")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("ไม่สามารถอ่านภาพจากกล้องได้")
        break

    cv2.imshow('C922 Webcam', frame)

    # กด 'q' เพื่อออก
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
