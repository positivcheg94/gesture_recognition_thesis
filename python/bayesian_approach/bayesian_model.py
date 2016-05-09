from collections import defaultdict
import pickle

import numpy as np


class DiffClasses(Exception):
	pass


class ClassificationMatrix:
	
	def __init__(self, probs_matrix_thresholded):
		self._probs = np.array(probs_matrix_thresholded)
	
	def classify(self, image):
		shape = image.shape
		img_r = image.reshape((-1,shape[-1]))
		res = np.zeros(shape[:2], dtype=np.bool)
		res_r = res.reshape(-1)
		for i in range(len(img_r)):
			res_r[i] = self._probs.item(*img_r[i])
		return res


class BayesianClasificatorResult:
	@staticmethod
	def load_from_file(file_name):
		obj =  pickle.load(open(file_name,'rb'))
		if not isinstance(obj,BayesianClasificatorResult):
			raise DiffClasses
		return obj

	def __init__(self, probs):
		self._probs = probs
	
	def __dumps__(self,file_name):
		pickle.dump(self,open(file_name,'wb'))
	
	def classify(self, image, treshold = 0):
		shape = image.shape
		img_r = image.reshape((-1,shape[-1]))
		res = np.zeros(shape[:2], dtype=np.bool)
		res_r = res.reshape(-1)
		for i in range(len(img_r)):
			res_r.itemset(i,self._probs.item(*img_r[i])>treshold)
		return res
		
	def classification_matrix(self, treshold):
		return ClassificationMatrix(self._probs > treshold)



class BayesianClasificator:
	@staticmethod
	def load_from_file(file_name):
		obj =  pickle.load(open(file_name,'rb'))
		if not isinstance(obj,BayesianClasificator):
			raise DiffClasses
		return obj

	def __init__(self, _color_space_bounds):
		self._color_space_bounds = np.array(_color_space_bounds)
		self._model = np.zeros(self._color_space_bounds)

	def __dumps__(self,file_name):
		pickle.dump(self,open(file_name,'wb'))
		
	def feed_image(self, image, mask):
		shape = image.shape
		assert(len(self._color_space_bounds) == shape[-1])
		res = defaultdict(int)
		for pixel, flag in zip(image.reshape((-1,shape[-1])),mask.reshape(-1)):
			if flag:
				self._model.itemset(*pixel,self._model.item(*pixel)+1)
		
	def get_result(self):
		m_elem = 1/np.max(self._model)
		return BayesianClasificatorResult(self._model*m_elem)


















	
