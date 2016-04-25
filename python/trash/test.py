import numpy as np
from skimage import color,feature,filters,segmentation,restoration
import cv2

cap = cv2.VideoCapture(2)

#CAP_INTELPERC_DEPTH_MAP
tmp = None
while cap.isOpened():
    # Capture frame-by-frame
    if cap.grab():
        ret, frame = cap.retrieve(cv2.CAP_INTELPERC_UVDEPTH_MAP)
        tmp = frame
        print(frame.shape)
    s = np.array(frame[:,:,2])
    s = s/255
    r = frame[:,:,0]+frame[:,:,1]+frame[:,:,2]
    r = (r - r.min())/(r.max()-r.min())
    #cv2.imshow('1',frame[:,:,0])
    #cv2.imshow('2',frame[:,:,1])
    #cv2.imshow('3',frame[:,:,2])
    cv2.imshow('default',r)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
