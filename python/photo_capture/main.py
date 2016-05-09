import sys

import cv2
import numpy as np

if len(sys.argv)>0:
	cap = cv2.VideoCapture(int(sys.argv[1]))
else:
	cap = cv2.VideoCapture(0)


capture = True

def create_tool_bar():
	cv2.namedWindow('toolbar')
	cv2.QT_PUSH_BUTTON


while (cap.isOpened()):
    ret, frame = cap.read()



    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
