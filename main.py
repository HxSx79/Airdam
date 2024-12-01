import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import numpy as np


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)


cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Load the YOLO11 model
model = YOLO("best.pt")

# Export the model
# model.export(format="engine")  # creates 'yolov11.engine'

# Load the exported TensorRT model
# trt_model = YOLO("best.engine")

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
# Open the video file (use video file or webcam, here using webcam)
cap = cv2.VideoCapture('Test_MY.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count = 0
cx1 = 150
cx2 = 180
offset = 8
inp = {}
enter = []
exp = {}
exitp = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (480, 856))

    # Run YOLO11 tracking on the frame, persisting tracks between frames
    results = model.track(frame, persist=True,conf=0.5)

    # Check if there are any boxes in the results
    if results[0].boxes is not None and results[0].boxes.id is not None:
        # Get the boxes (x, y, w, h), class IDs, track IDs, and confidences
        boxes = results[0].boxes.xyxy.int().cpu().tolist()  # Bounding boxes
        class_ids = results[0].boxes.cls.int().cpu().tolist()  # Class IDs
        track_ids = results[0].boxes.id.int().cpu().tolist()  # Track IDs
        confidences = results[0].boxes.conf.cpu().tolist()  # Confidence score

        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
            c = 'Tesla_MY'
            x1, y1, x2, y2 = box
            cx=int(x1+x2)//2
            cy=int(y1+y2)//2
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cvzone.putTextRect(frame, f'{track_id}', (x1, y2), .5, 1)
            cvzone.putTextRect(frame, f'{c}', (x1, y1), .5, 1)
            if cx2<(cx+offset) and cx2>(cx-offset):
                inp[track_id]=(cx,cy)
            if track_id in inp:
                if cx1<(cx+offset) and cx1>(cx-offset):

                    #cv2.circle(frame,(cx,cy),4,(255,0,0),-1)
                    #cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    #cvzone.putTextRect(frame, f'{track_id}', (x1, y2), 1, 1)
                    #cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
                    if enter.count(track_id)==0:
                        enter.append(track_id)

    cv2.line(frame,(150,175),(150,720),(0,0,255),2)
    cv2.line(frame,(180,175),(180,720),(255,0,255),2)
    enterp=len(enter)
    cvzone.putTextRect(frame, f'MY_PARTS:{enterp}', (50, 50), 1, 1)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
