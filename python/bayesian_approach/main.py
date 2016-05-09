import sys

import cv2
import numpy as np

from skimage.viewer import ImageViewer

from bayesian_model import BayesianClasificatorResult

if len(sys.argv)>1:
	cap = cv2.VideoCapture(int(sys.argv[1]))
else:
	cap = cv2.VideoCapture(0)

bres = BayesianClasificatorResult.load_from_file('trained.clr')
cm = bres.classification_matrix(0.1)


while cap.isOpened():
	ret, frame = cap.read()
	classified = cm.classify(frame)
	
	viewer = ImageViewer(classified)
	viewer.show()
	print('lalala')
	#cv2.imshow('webcam',np.array(classified,dtype=np.uint8))

	k = cv2.waitKey(0) & 0xff
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()
