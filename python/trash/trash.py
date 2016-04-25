import numpy as np
from skimage import color,feature,filters,segmentation,restoration,draw

CONST_CIRCLE_COLOR = np.array([255,0,0])

def visualize_features(image,features,method='dog', mode = 'circle', color = CONST_CIRCLE_COLOR):
	if method == 'dog' or method == 'log':
		r_mul = np.sqrt(2)
	else:
		r_mul = np.sqrt(1)
	if mode == 'circle_perimeter':
		figure = draw.circle_perimeter
	else:
		figure = draw.circle
	ret_image = np.array(image)
	shape = ret_image.shape[:-1]
	for dog_feature in features:
		x,y,sigma = dog_feature
		x,y = int(x),int(y)
		rr,cc = figure(x,y,int(r_mul*sigma),shape=shape)
		ret_image[rr,cc] = CONST_CIRCLE_COLOR
	return ret_image
