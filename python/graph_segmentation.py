import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while (1):
    ret, frame = cap.read()
    seg = cv2.ximgproc.segmentation.createGraphSegmentation()
    img = seg.processImage(frame)
    cv2.imshow('default', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
