import numpy as np
import cv2

cap = cv2.VideoCapture(0)

haar = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')


while True:
    _,img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detects = haar.detectMultiScale(gray,3,3)

    centers = []
    for (x,y,w,h) in detects:
        centers.append((x+int(w/2),y+int(h/2)))
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    for center in centers:
        cv2.circle(img,center,3,(255,255,255))

        

    cv2.imshow('img',img)
    key = cv2.waitKey(20)
    if (key != -1):
        print(key)
    if (key == 27):
        break
cv2.destroyAllWindows()