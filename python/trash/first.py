from time import time

from skimage import exposure

import cv2


def timeit(f):
    def wrap(*args,**kwargs):
        start = time()
        ret = f(*args,**kwargs)
        print('%s function took %0.3f ms' % (f.__name__, (time()-start)*1000.0))
        return ret
    return wrap

cap = cv2.VideoCapture(0)

adjust_gamma = timeit(exposure.adjust_gamma)
equalize_adapthist = timeit(exposure.equalize_adapthist)

while(True):
    _,frame = cap.read()

    start = time()
    adjusted = equalize_adapthist(frame,2)

    cv2.imshow('frame', frame)
    cv2.imshow('adjusted',adjusted)

    if cv2.waitKey(20) & 0xFF == 27:
        break

