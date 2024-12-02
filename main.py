from ultralytics import YOLO
import cv2
import math 
# start webcam
camera_id = "/dev/video1"
cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
cap.set(3, 640)
cap.set(4, 640)

# model
model = YOLO("best.pt")

# object classes
classNames = ["Airdam", "Clip_OK", "Clip_NOK"]

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # object details
            ycl = y1 - 10
            org = [x1, ycl]
            ycf = y2 + 20
            orgc = [x1,ycf]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            color = (255, 0, 0)
            thickness = 1

            conf = str(confidence)

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
            cv2.putText(img, conf, orgc, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
