import os
import random as r
import time as t

import numpy as np
import cv2

from utils.slideshow import slideshow

takes = slideshow([ 
	f"./input/cut_pics/{o}" 
	for o in os.listdir("./input/cut_pics")
	if o.startswith("notes_") and o.endswith("2.jpg")
])
cur_take = cv2.imread("./input/cut_pics/calibration_card.jpg")
next_take = cur_take

h, w, _ = cur_take.shape

crop = np.zeros((w, w, 3), dtype= np.uint8)

off = 0
OFFSET_GAIN = 25
h_off_jit = 0.
w_off_jit = 0.

finalizing = False

while True:
	# TODO: pre-fill crop with bg
	# TODO: add jitter as additional offset

	if off > h:
		cur_take = next_take
		off -= h

	crop_cur = cur_take[off:off+w]
	hh = crop_cur.shape[0]

	crop[:hh, :, :] = crop_cur
	if hh < w:
		if id(cur_take) == id(next_take):
			next_take = next(takes, None)
		if next_take is None:
			if not finalizing:
				next_take = cv2.imread("./input/cut_pics/calibration_bg.jpg")
				finalizing = True
			else:
				break
		crop[hh:, :, :] = next_take[:w-hh, :, :]

	out_crop = cv2.resize(crop, (w // 2, w // 2))

	cv2.imshow("crop", out_crop)
	cv2.waitKey(1000 // 60)

	off += OFFSET_GAIN

cv2.destroyAllWindows()
print("Finished")

