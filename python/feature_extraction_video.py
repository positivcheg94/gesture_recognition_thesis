import numpy as np

import cv2


cap = cv2.VideoCapture(0)

while cap.isOpened():
	# Capture frame-by-frame
	ret, frame = cap.read()
	#gray = color.rgb2gray(frame)

	gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.08,edgeThreshold=10,sigma=1)
	kp = sift.detect(gray,None)

	cv2.drawKeypoints(frame,kp,frame,color=[255,0,0],flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	cv2.imshow('default',frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
