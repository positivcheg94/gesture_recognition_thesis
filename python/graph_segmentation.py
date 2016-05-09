import sys

import cv2
import numpy as np

if len(sys.argv)>0:
	cap = cv2.VideoCapture(int(sys.argv[1]))
else:
	cap = cv2.VideoCapture(0)

while (cap.isOpened()):
    ret, frame = cap.read()
    seg = cv2.ximgproc.segmentation.createGraphSegmentation()
    img = seg.processImage(frame)
    cv2.imshow('default', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
