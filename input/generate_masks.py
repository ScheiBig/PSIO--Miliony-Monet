import scipy.ndimage # type: ignore [import-untyped] # :(
import skimage as ski
import numpy as np
from matplotlib import pyplot as plt
import os
from utils import ansi

for o in os.listdir("./input/masks"):
	if not o.endswith(".jpg"):
		continue

	os.makedirs(f"./input/masks/{o[:-3]}", exist_ok=True)

	base: np.ndarray = ski.io.imread(f"./input/masks/{o}")
	new_size = int(np.hypot(base.shape[0], base.shape[1]))
	new_base: np.ndarray = np.zeros((new_size, new_size, 3), dtype=np.uint8)

	print(new_size)

	h, w = base.shape

	h_o = int((new_base.shape[0] - h) / 2)
	w_o = int((new_base.shape[1] - w) / 2)

	for i in range(360):

		ansi.progress(i // 4, 89, returns=True, label= f"Eksport {o}")

		new_mask: np.ndarray = new_base.copy()
		new_mask[h_o : h_o + h, w_o : w_o + w, :] = base.copy()

		new_mask = scipy.ndimage.rotate(new_mask, 360 - i, reshape=True)

		b_mask: np.ndarray = np.max(new_mask, axis= 2)
		new_mask = new_mask[~np.all(b_mask == 0, axis=1), :, :]

		b_mask = b_mask[~np.all(b_mask == 0, axis=1)]
		new_mask = new_mask[:, ~np.all(b_mask == 0, axis=0), :]

		ski.io.imsave(f"./input/masks/{o[:-3]}/{i}.jpg", new_mask)

	
	
	