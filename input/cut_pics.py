import os

import skimage as ski
import numpy as np

import utils.ansi as ansi

list_dir = [ i for i in os.listdir("./input/raw_pics") ]
print()

for i, im in enumerate(list_dir):
	prog = int(i / (len(list_dir) - 1) * 40)

	ansi.progress(
		prog, 40, 
		returns= True, 
		label= f"cutting: {im:<20}", 
		final= f"done cutting {str():<16}"
	)

	img: np.ndarray = ski.io.imread(f"./input/raw_pics/{im}")
	new_img: np.ndarray = img[80:-50, 60:-90].copy()

	ski.io.imsave(f"./input/cut_pics/{im}", new_img)
