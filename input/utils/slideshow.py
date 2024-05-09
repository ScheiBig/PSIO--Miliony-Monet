import collections.abc as col_abc

import cv2
import numpy as np

__all__ = [
	"slideshow"
]

def slideshow(of: col_abc.Iterator[str]) -> col_abc.Generator[np.ndarray, None, None]:
	'''
	Accepts collection of paths to images, and returns lazy iterator over images \
		loaded with ``cv2``.
	:param of: Collection of paths; it should only contain images (that ``cv2`` na read) \
		and be already sorted in preferred order.
	'''

	for o in of:
		yield cv2.imread(o)
