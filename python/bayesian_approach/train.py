import os
from collections import defaultdict
import re

import numpy as np
import cv2

from bayesian_model import BayesianClasificator, BayesianClasificatorResult

photo_dir_name = 'photos_and_masks'
photo_dit_path = './{}/'.format(photo_dir_name)

files = os.listdir(photo_dit_path)

samples = defaultdict(dict)

white = np.array([255,255,255])

for (n,t,ext),full_name in zip(map(lambda x:re.split('[._]',x), files), files):
	if t == 'm':
		image = cv2.imread(photo_dit_path+full_name)
		res = np.all(image==white,axis=2)
	else:
		res = cv2.imread(photo_dit_path+full_name)
	samples[int(n)][t] = res

bc = BayesianClasificator([256,256,256])
for k in samples:
	bc.feed_image(samples[k]['i'],samples[k]['m'])
res = bc.get_result().__dumps__('trained.clr')
