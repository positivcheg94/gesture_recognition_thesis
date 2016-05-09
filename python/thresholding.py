import numpy as np

import cv2

from skimage import filters


cap = cv2.VideoCapture(0)

while cap.isOpened():
	ret, frame = cap.read()

	blur = cv2.GaussianBlur(frame,(11,11),0)
	gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,6)

	cv2.imshow('1',thresh)

	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
