import numpy as np

import cv2

cap = cv2.VideoCapture(0)

while cap.isOpened():
	# Capture frame-by-frame
	ret, frame = cap.read()
	frame = np.fliplr(frame)
	f_shape = frame.shape

	gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	sift_extractor = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.04,edgeThreshold=10,sigma=1)
	sift_kp = sift_extractor.detect(gray, None)
	sift_frame = np.array(frame)
	cv2.drawKeypoints(frame,sift_kp,sift_frame,color=[255,0,0],flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	cv2.imshow('sift',sift_frame)

	"""
	star_detector_extractor = cv2.xfeatures2d.StarDetector_create(maxSize=20,responseThreshold=30,lineThresholdProjected=20,lineThresholdBinarized=15,suppressNonmaxSize=5)
	star_detector_kp = star_detector_extractor.detect(gray, None)
	star_detector_frame = np.array(frame)
	cv2.drawKeypoints(frame,star_detector_kp,star_detector_frame,color=[255,0,0],flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	cv2.imshow('star_detector',star_detector_frame)
	"""
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
