import os
import skimage as ski
import numpy as np
import matplotlib.pyplot as plt

for im in os.listdir("./input/raw_pics"):
	img = ski.io.imread(f"./input/raw_pics/{im}")
	h, w, _ = img.shape
	if h < w:
		print(f"rotating: {im}")
		img = np.rot90(img, -1)
		ski.io.imsave(f"./input/raw_pics/{im}", img)
