import os
import random as r
import time as t
import subprocess as sps

import numpy as np
import cv2

from utils import *

r.seed("Chcemy losowo, ale przewidywalnie")

takes = slideshow([ 
	f"./input/cut_pics/{o}" 
	for o in os.listdir("./input/cut_pics")
	if o.startswith("notes_") and not o.endswith("org.jpg") and not o.endswith("unused.jpg")
])

cur_take = cv2.imread("./input/cut_pics/calibration_card.jpg")
next_take = cur_take
bg_take = cv2.imread("./input/cut_pics/calibration_bg.jpg")

h, w, _ = cur_take.shape

crop = np.zeros((w, w, 3), dtype= np.uint)

off = 0
OFFSET_GAIN = 25
h_off_jit = 0.
MAX_H_JIT = 10
w_off_jit = 0.
MAX_W_JIT = 30
off_bg = (h - w) // 2

finalizing = False

writer = cv2.VideoWriter(
	"./input/notes_on_belt.mkv",
	cv2.VideoWriter.fourcc(*"h264"),
	fps= 30,
	frameSize= (w // 2, w // 2),
	isColor= True
)

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
	crop = bg_take[off_bg:-off_bg, :, :].copy() # type: ignore [assignment] # type inheritance allows this

	if off > h:
		cur_take = next_take
		off -= h

	crop_cur = cur_take[off:off+w]
	hh = crop_cur.shape[0]
	
	crop[:hh, w_off["x_"], :] = crop_cur[:, w_off["_x"], :]
	if hh < w:
		if id(cur_take) == id(next_take):
			next_take = next(takes, None) # type: ignore [arg-type] # None to prevent throwing
		if next_take is None:
			if not finalizing:
				next_take = bg_take
				finalizing = True
			else:
				break
		crop[hh:, w_off["x_"], :] = next_take[:w - hh, w_off["_x"], :]

	out_crop = cv2.resize(crop, (w // 2, w // 2))

	ansi.progress(i // 100, 67, label=f"{i:7} of 6715", returns=True)

	# cv2.imshow("crop", out_crop)
	# cv2.waitKey(1000 // 30)
	writer.write(out_crop)

	off += OFFSET_GAIN + h_off


writer.release()
cv2.destroyAllWindows()
print("Finished rendering")

sps.call([
	"ffmpeg",
	"-i", 
		"./input/notes_on_belt.mkv",
	"-vcodec",
	 	"libx264",
	"-b:v",
		"2M",
	"-minrate",
		"2M",
	"-maxrate",
		"3M",
	"-bufsize",
		"4M",
	"-vf",
		"scale=858:858",
	"-r",
		"30",
	"-y",
	"./input/notes_on_belt_sm.mkv"
])

os.remove("./input/notes_on_belt.mkv")
os.rename(
	"./input/notes_on_belt_sm.mkv",
	"./input/notes_on_belt.mkv"
)
