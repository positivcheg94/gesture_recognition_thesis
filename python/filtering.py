import cv2
import numpy as np

class NumChannelsMismatch:
	pass

class image_show:

	def __init__(self, channels,names_map):
		self._channels = channels
		self._names_map = list(names_map)

	def show_image(self, image):
		if image.shape[2] != self._channels:
			raise NumChannelsMismatch()
		for i in range(self._channels):
			cv2.imshow(self._names_map[i],image[:,:,i])


class image_processor:

	def __init__(self,color_filter = None, erode_kernel = None, dilating_kernel=None, max_iterations = 1):
		self._color_filter = color_filter
		self._erode_kernel = erode_kernel
		self._dilating_kernel = dilating_kernel

		self._intermadiate_image = None

	def __filter_color__(self):


	def __erode__(self):
		self._intermadiate_image = cv2.erode(self._intermadiate_image,self._erode_kernel, iterations=1)

	def __dilate__(self):
		self._intermadiate_image = cv2.dilate(self._intermadiate_image,self._dilating_kernel)

	def process_image(self, image):
		pass

bgr_channels = ['blue','green','red']
hsv_channels = ['hue','saturation','value']

cap = cv2.VideoCapture(0)

while cap.isOpened():
	ret,img = cap.read()
	hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

	img_show = image_show(3, hsv_channels)
	img_show.show_image(hsv)

	if cv2.waitKey(1) & 0xFF == 27:
		break
