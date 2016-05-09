import sys

import cv2
import numpy as np

if len(sys.argv)>0:
	cap = cv2.VideoCapture(int(sys.argv[1]))
else:
	cap = cv2.VideoCapture(0)

while (cap.isOpened()):
	# Capture frame-by-frame
	ret, frame = cap.read()
	#gray = color.rgb2gray(frame)

	gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	detector = cv2.xfeatures2d.SURF_create(hessianThreshold=250, nOctaves=1, nOctaveLayers=5 , extended=True, upright=True)

	kp = detector.detect(gray,None)

	cv2.drawKeypoints(frame,kp,frame,color=[255,0,0]
	#,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
	)
	cv2.imshow('default',frame)

	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
