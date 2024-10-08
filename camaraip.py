import numpy as np
import cv2

cap = cv2.VideoCapture()
cap.open('rtsp://admin:leteragosl2024@192.168.1.134:554/Streaming/channels/2/')

while True:
    ret, frame = cap.read()
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()