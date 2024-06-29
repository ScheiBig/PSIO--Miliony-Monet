import cv2
import numpy as np
import skimage as ski

def threshold_mask(img: np.ndarray, threshold: int) -> np.ndarray:
	'''
	Returns mask that removes background from image using threshold.
	Mask is pre-processed to remove most small particles and remove some tiny holes.

	:param img: Grayscale (single channel) image.
	:param threshold: Threshold value, that is in range 0..255.
	:return: Mask that allows foreground elements, that are brighter than ``threshold``.
	Mask is not binary (255 instead of 1).
	'''
	
	mask = np.uint8(img > threshold) * 255
	opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ski.morphology.disk(3))
	return cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, ski.morphology.disk(3))
	