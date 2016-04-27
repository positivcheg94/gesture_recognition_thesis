import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while (1):
    ret, frame = cap.read()

    c_image = cv2.Canny(frame,threshold1=70,threshold2=200,apertureSize=3,L2gradient=False)

    wtf, contours, hierarchy = cv2.findContours(c_image, mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image = frame, contours = contours, contourIdx = -1, color=(255,0,0), thickness = 1)
    cv2.imshow('default', frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
