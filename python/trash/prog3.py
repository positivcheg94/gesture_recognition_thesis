import numpy as np
import cv2

lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([2, 255, 255], dtype = "uint8")

cap = cv2.VideoCapture(1)

while cap.isOpened():
	# Capture frame-by-frame
	ret, frame = cap.read()
	rgb_frame = np.fliplr(frame)

	hsv_frame = cv2.cvtColor(rgb_frame,cv2.COLOR_BGR2HSV)
	gray_frame = cv2.cvtColor(rgb_frame,cv2.COLOR_BGR2GRAY)
	ycrcb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2YCR_CB)

	"""
	skin_ycrcb_mint = np.array((0, 133, 77))
	skin_ycrcb_maxt = np.array((255, 173, 127))
	skin_ycrcb_mask = cv2.inRange(ycrcb_frame, skin_ycrcb_mint, skin_ycrcb_maxt)

	skin_ycrcb = cv2.bitwise_and(rgb_frame, rgb_frame, mask = skin_ycrcb_mask)
	cv2.imshow('skin_ycrcb',skin_ycrcb)
	"""

	hsv_skin_mask = cv2.inRange(hsv_frame, lower, upper)
	#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
	#hsv_skin_mask = cv2.erode(hsv_skin_mask, kernel, iterations = 2)
	#hsv_skin_mask = cv2.dilate(hsv_skin_mask, kernel, iterations = 2)

	hsv_skin_mask = cv2.GaussianBlur(hsv_skin_mask, (3, 3), 0)
	hsv_skin = cv2.bitwise_and(rgb_frame, rgb_frame, mask = hsv_skin_mask)

	cv2.imshow('hsv_skin',hsv_skin)



	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
