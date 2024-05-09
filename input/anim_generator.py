import os
import random as r
import time as t

import numpy as np
import cv2

from utils import ansi, ints, offsets
from utils.pid import pid
from utils.slideshow import slideshow

r.seed("Chcemy losowo, ale przewidywalnie")

takes = slideshow([ 
	f"./input/cut_pics/{o}" 
	for o in os.listdir("./input/cut_pics")
	if o.startswith("notes_")# and o.endswith("2.jpg")
])
cur_take = cv2.imread("./input/cut_pics/calibration_card.jpg")
next_take = cur_take
bg_take = cv2.imread("./input/cut_pics/calibration_bg.jpg")

h, w, _ = cur_take.shape

crop = np.zeros((w, w, 3), dtype= np.uint8)

off = 0
OFFSET_GAIN = 25
h_off_jit = 0.
MAX_H_JIT = 10
w_off_jit = 0.
MAX_W_JIT = 30
off_bg = (h - w) // 2

finalizing = False

print()

h_jit = pid(set_point= 0, p= 0.02, i= 0, d= 0)
w_jit = pid(set_point= 0, p= 0.02, i= 0, d= 0.1)
INTs = ints()
for i in INTs:
	match i % 200:
		case 100:
			h_jit.change_param(set_point= r.random() * MAX_H_JIT)
			w_jit.change_param(set_point= r.random() * MAX_W_JIT)
		case 0:
			h_jit.change_param(set_point= r.random() * -MAX_H_JIT)
			w_jit.change_param(set_point= r.random() * -MAX_W_JIT)

	h_off_jit += h_jit.update(h_off_jit)
	w_off_jit += w_jit.update(w_off_jit)

	h_off = int(h_off_jit)
	w_off = offsets(int(w_off_jit))
	crop = bg_take[off_bg:-off_bg, :, :].copy()

	# TODO: add jitter as additional offset

	if off > h:
		cur_take = next_take
		off -= h

	crop_cur = cur_take[off:off+w]
	hh = crop_cur.shape[0]
	
	crop[:hh, w_off["x_"], :] = crop_cur[:, w_off["_x"], :]
	if hh < w:
		if id(cur_take) == id(next_take):
			next_take = next(takes, None)
		if next_take is None:
			if not finalizing:
				next_take = bg_take
				finalizing = True
			else:
				break
		crop[hh:, w_off["x_"], :] = next_take[:w - hh, w_off["_x"], :]

	out_crop = cv2.resize(crop, (w // 2, w // 2))

	ansi.progress(i // 100, 67, label=f"{i:7} of 6736", returns= True)

	cv2.imshow("crop", out_crop)
	cv2.waitKey(1000 // 30)

	off += OFFSET_GAIN + h_off

cv2.destroyAllWindows()
print("Finished")
