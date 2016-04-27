import cv2
import numpy as np

from skimage import color

def strange_map(f_sequence, *args):
	for function in f_sequence:
		yield function(*args)

class NumChannelsMismatch:
	pass

class ColorConverter:
	opencv = cv2.__dict__
	opencv_color_mask = 'COLOR_{}2{}'
	skimage_color = color.__dict__
	skimage_color_mask = '{}2{}'
	@staticmethod
	def __opencv__(f,s):
		return lambda x: cv2.cvtColor(x,opencv[opencv_color_mask.format(f.upper(),s.upper)])
	@staticmethod
	def __skimage__(f,s):
		return skimage_color[skimage_color_mask.format(f.lower(),s.lower()]

class ImageShow:
	rgb = ['red','green','red']
	bgr = ['blue','green','blue']
	hsv = ['hue','saturation','value']

	def __init__(self, channels,names_map):
		self._channels = channels
		self._names_map = list(names_map)

	def show_image(self, image, separated=True):
		if image.shape[2] != self._channels:
			raise NumChannelsMismatch
		for i in range(self._channels):
			cv2.imshow(self._names_map[i],image[:,:,i])

class ColorFilter:

	def __init__(self,channels, mask_dtype = np.bool):
		self._channels = channels
		self._filter_sequence = set([])
		self._mask_dtype = mask_dtype

	def append_filter(self, color_filter):
		if color_filter.__code__.co_argcount != self._channels:
			raise NumChannelsMismatch
		self._filter_sequence.add(color_filter)

	def pop_filter(self, color_filter = None):
		if color_filter:
			try:
				self._filter_sequence.remove(color_filter)
			finally:
				pass
		else:
			self._filter_sequence.pop()

	def __call__(self, image):
		if image.shape[2] != self._channels:
			raise NumChannelsMismatch
		mask = np.zeros(image.shape,dtype=self._mask_dtype)
		plain_mask = filtered_image.reshape((-1,filtered_image.shape[-1]))
		for i in range(len(plain_mask)):
			plain_mask[i] = all(strange_map(self._filter_sequence,color))
		return mask



class image_processor:

	def __init__(self,color_filter = None, erode_kernel = None, dilating_kernel=None, max_iterations = 1):
		self._color_filter = color_filter
		self._erode_kernel = erode_kernel
		self._dilating_kernel = dilating_kernel

		self._intermadiate_image = None

	def __filter_color__(self):
		pass

	def __erode__(self):
		self._intermadiate_image = cv2.erode(self._intermadiate_image,self._erode_kernel, iterations=1)

	def __dilate__(self):
		self._intermadiate_image = cv2.dilate(self._intermadiate_image,self._dilating_kernel)

	def process_image(self, image):
		pass



cap = cv2.VideoCapture(0)

while cap.isOpened():
	ret,img = cap.read()
	hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

	img_show = ImageShow(3, ImageShow.hsv)
	img_show.show_image(hsv)

	if cv2.waitKey(1) & 0xFF == 27:
		break
