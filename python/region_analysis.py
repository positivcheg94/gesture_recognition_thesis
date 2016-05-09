import sys

import cv2
import numpy as np

if len(sys.argv)>0:
	cap = cv2.VideoCapture(int(sys.argv[1]))
else:
	cap = cv2.VideoCapture(0)

mr = (270,370),(190,290),(255,255,255)


top_left_corner = (0,0),(100,100)
top_right_corner = (540,0),(640,100)

while (cap.isOpened()):
    _,img = cap.read()

    region = img[mr[1][0]:mr[1][1],mr[0][0]:mr[0][1]]
    img[top_left_corner[0][0]:top_left_corner[1][0], top_left_corner[0][1]:top_left_corner[1][1]] = region

    hsv_color = cv2.cvtColor(region,cv2.COLOR_BGR2HSV)
    color_min = np.min(hsv_color,axis=(0,1))
    color_max = np.max(hsv_color, axis=(0, 1))

    print('colors are', color_min, color_max)


    cv2.rectangle(img,(mr[0][0],mr[1][0]),(mr[0][1],mr[1][1]),mr[2],2)
    #cv2.rectangle(img, top_right_corner[0], top_right_corner[1], color_mean, -1)
    cv2.imshow('img',img)
    key = cv2.waitKey(20)
    if cv2.waitKey(20) & 0xFF == 27:
        break
