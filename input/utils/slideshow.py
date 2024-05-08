import collections.abc as col_abc

import cv2
import numpy as np

def slideshow(of: list[str]) -> col_abc.Generator[np.ndarray, None, None]:
	for o in of:
		yield cv2.imread(o)
