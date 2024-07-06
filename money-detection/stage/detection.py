import math
import time
from typing import Literal, Tuple, TypedDict, cast
import cv2
import cv2.typing as cv2_t
import numpy as np
from stage import calibration
from abc import abstractmethod, ABC
import pytesseract
import skimage as ski

AVG_ELEMENT_DROP_PTC = 0.05
AVG_VALID_AVG_MAX = 24
GAUSS_K_SIZE = 5

class detected_banknote(ABC):

	def __init__(self,
		value: int,
		rect: cv2_t.RotatedRect,
		denomination: cv2_t.RotatedRect,
		side: Literal["front", "back"],
		bank_name: tuple[cv2_t.RotatedRect, int] | None
	) -> None:
		
		self.value: int = value
		self.rect: cv2_t.RotatedRect = rect
		self.denomination: cv2_t.RotatedRect = denomination
		self.side: Literal["front", "back"] = side
		self.bank_name: tuple[cv2_t.RotatedRect, int] | None = bank_name
		pass

	@abstractmethod
	def draw(self, img: np.ndarray) -> np.ndarray:
		'''
		Draws this banknote features onto provided image.

		:param img: image to draw banknote to.
		:return: ``img`` after modifications.
		'''
		return
	
	@abstractmethod
	def confidence(self) -> float:
		'''
		Calculates confidence of detection.

		:return: % of confidence that label was assigned correctly.
		'''
		return


class banknote_front(detected_banknote):

	def __init__(self,
		value: int,
		rect: cv2_t.RotatedRect,
		denomination: cv2_t.RotatedRect,
		bank_name: tuple[cv2_t.RotatedRect, int] | None,
		symbol: np.ndarray | None,
	):
		super().__init__(value, rect, denomination, "front", bank_name)

		self.symbol: np.ndarray | None = symbol

	def draw(self, img: np.ndarray) -> np.ndarray:
		
		b_x, b_y, _, _ = cv2.boundingRect(self.rect)

		cv2.drawContours(
			img,
			np.int_(cv2.boxPoints(self.rect)),
			-1,
			(0, 255, 0),
			3
		)
		cv2.putText(
			img,
			f"{self.value}zł Awe: {self.confidence()}%",
			(b_x + 4, b_y + 4),
			cv2.FONT_HERSHEY_SIMPLEX,
			1,
			(64, 255, 64),
			1
		)
		cv2.putText(
			img,
			f"NBP: {self.bank_name[1]}/18 liter",
			(b_x + 4, b_y + 20),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.75,
			(64, 255, 64),
			1
		)

		cv2.drawContours(
			img,
			np.int_(cv2.boxPoints(self.denomination)),
			-1,
			(255, 255, 0),
			2
		)
		cv2.drawContours(
			img,
			np.int_(cv2.boxPoints(self.bank_name[0])),
			-1,
			(255, 255, 0),
			2
		)
		cv2.drawContours(
			img,
			self.symbol,
			-1,
			(255, 255, 0),
			2
		)

		return img


	def confidence(self) -> float:
		conf = 0.6
		if self.symbol is not None:
			conf += 0.22
		if self.bank_name is not None:
			conf += self.bank_name[1] * 0.1

		return conf

class banknote_back(detected_banknote):

	def __init__(self,
		value: int,
		rect: cv2_t.RotatedRect,
		denomination: cv2_t.RotatedRect,
		bank_name: tuple[cv2_t.RotatedRect, int] | None,
		bank_symbol: cv2_t.RotatedRect | None,
	):
		super().__init__(value, rect, denomination, "back", bank_name)

		self.bank_symbol: cv2_t.RotatedRect | None = bank_symbol

	def draw(self, img: np.ndarray) -> np.ndarray:
		b_x, b_y, _, _ = cv2.boundingRect(self.rect)

		cv2.drawContours(
			img,
			np.int_(cv2.boxPoints(self.rect)),
			-1,
			(0, 255, 0),
			3
		)
		cv2.putText(
			img,
			f"{self.value}zł Awe: {self.confidence()}%",
			(b_x + 4, b_y + 4),
			cv2.FONT_HERSHEY_SIMPLEX,
			1,
			(64, 255, 64),
			1
		)
		cv2.putText(
			img,
			f"NBP: {self.bank_name[1]}/18 liter",
			(b_x + 4, b_y + 20),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.75,
			(64, 255, 64),
			1
		)

		cv2.drawContours(
			img,
			np.int_(cv2.boxPoints(self.denomination)),
			-1,
			(255, 255, 0),
			2
		)
		cv2.drawContours(
			img,
			np.int_(cv2.boxPoints(self.bank_name[0])),
			-1,
			(255, 255, 0),
			2
		)
		cv2.drawContours(
			img,
			self.bank_symbol,
			-1,
			(255, 255, 0),
			2
		)

		return img

	def confidence(self) -> float:
		conf = 0.6
		if self.bank_symbol is not None:
			conf += 0.22
		if self.bank_name is not None:
			conf += self.bank_name[1] * 0.1

		return conf

def detect_banknotes(
	img: np.ndarray,
	mask: np.ndarray,
	rects: list[cv2_t.RotatedRect]
) -> tuple[list[detected_banknote, cv2_t.RotatedRect]]:
	
	ret: tuple[list[detected_banknote, cv2_t.RotatedRect]] = ([], [])

	for rect in rects:
		(r_x, r_y), (r_w, r_h), r_a = rect
		r_x, r_y, r_w, r_h = int(r_x), int(r_y), int(r_w), int(r_h)
		bbox = cv2.boundingRect(np.int_(cv2.boxPoints(rect)))
		b_x, b_y, b_w, b_h = cast(list[int], bbox)
		
		if r_w > r_h:
			r_h, r_w = r_w, r_h
			r_a = (r_a + 90) % 360

		# create new image to paste and rotate banknote
		side = int(math.hypot(r_w, r_h)) + 2
		banknote = np.zeros((side, side, 3), dtype=np.uint8)
		
		# copy banknote ROI into center of new image
		off_x = (side - b_w) // 2
		off_y = (side - b_h) // 2
		banknote[off_y : off_y + b_h, off_x : off_x + b_w, :] = img[b_y : b_y + b_h, b_x : b_x + b_w, :]

		# rotate banknote into vertical position
		r_mat = cv2.getRotationMatrix2D((side // 2, side // 2), r_a, 1)
		banknote = cv2.warpAffine(banknote, r_mat, (side, side))
		cv2.imshow(f"{time.time()}", banknote)
		# remove outside bands
		off_x = (side - r_w) // 2
		off_y = (side - r_h) // 2
		banknote = banknote[off_y : off_y + r_h, off_x : off_x + r_w, :]

		# try to detect in each orientation

		note = _detect_front(
			banknote,
			((r_x, r_y), (r_w, r_h), r_a)
		)
		if note is not None:
			ret[0].append(note)
			continue

		note = _detect_back(
			np.rot90(banknote, 1),
			((r_x, r_y), (r_w, r_h), r_a + 90)
		)
		if note is not None:
			ret[0].append(note)
			continue

		note = _detect_front(
			np.rot90(banknote, 2),
			((r_x, r_y), (r_w, r_h), r_a)
		)
		if note is not None:
			ret[0].append(note)
			continue

		note = _detect_back(
			np.rot90(banknote, 3),
			((r_x, r_y), (r_w, r_h), r_a + 90)
		)
		if note is not None:
			ret[0].append(note)
			continue

	return ret

def _rotate(point: tuple[int, int], origin: tuple[int, int], angle: float):
	"""
	Rotate a point counterclockwise by a given angle around a given origin.
	origin
	:param point: point to rotate.
	:param origin: origin point of rotation.
	:param angle: angle of rotation, in degrees.
	:return: new position of point after rotation.
	"""
	angle = math.radians(angle)
	o_x, o_y = origin
	p_x, p_y = point
	q_x = o_x + math.cos(angle) * (p_x - o_x) - math.sin(angle) * (p_y - o_y)
	q_y = o_y + math.sin(angle) * (p_x - o_x) + math.cos(angle) * (p_y - o_y)
	return int(q_x), int(q_y)

rotatedIntRect = tuple[tuple[int, int], tuple[int, int], float]

def _detect_front(banknote: np.ndarray, og_rect: rotatedIntRect) -> banknote_front | None:
	(r_x, r_y), (r_w, r_h), r_a = og_rect
	return

def _detect_back(banknote: np.ndarray, og_rect: rotatedIntRect) -> banknote_back | None:
	(r_x, r_y), (r_w, r_h), r_a = og_rect

	c = calibration.calibrated_size["0"]
	H = int(c * 3.5)
	W = int(c * 3)

	den = banknote[ :H , -W:, :]

	den = cv2.cvtColor(den, cv2.COLOR_BGR2GRAY)
	thr: int = min(ski.filters.threshold_multiotsu(den, 3)) - 10
	den: np.ndarray = np.uint8(den < thr) * 255
	den = cv2.resize(den, (den.shape[1] * 2, den.shape[0] * 2), interpolation= cv2.INTER_LINEAR)
	den = cv2.morphologyEx(den, cv2.MORPH_OPEN, ski.morphology.disk(1))

	print("Detected denomination:", pytesseract.image_to_string(den))

	# cv2.imshow(f"{time.time()}", banknote)
	cv2.imshow(f"{time.time()}", den)


# def detect_banknotes(
# 	img: np.ndarray,
# 	mask: np.ndarray,
# 	rects: list[cv2_t.RotatedRect]
# ):

# 	ret: list[tuple[tuple[float, float], str]] = []

# 	for rect in rects:
# 		(x, y), (h, w), a = rect
# 		if w < h:
# 			h, w = w, h
# 		else:
# 			a = (a + 90) % 360
		
# 		a_alt = (a + 180) % 360

# 		vals = list(calibration.calibrated_size.copy().items())
# 		vals.sort(key= lambda p: abs(p[1][0] - w) + abs(p[1][1] - h))

# 		attempts: list[tuple[float, str]] = []

# 		for val, _ in vals[:3]:
# 			m = _get_mask(val, "front", int(a))
# 			c0 = _check_if_mask_matches(img, mask, np.int_((x, y)), int(a), m)
# 			m = _get_mask(val, "back", int(a))
# 			c1 = _check_if_mask_matches(img, mask, np.int_((x, y)), int(a), m)
# 			m = _get_mask(val, "front", int(a_alt))
# 			c2 = _check_if_mask_matches(img, mask, np.int_((x, y)), int(a_alt), m)
# 			m = _get_mask(val, "back", int(a_alt))
# 			c3 = _check_if_mask_matches(img, mask, np.int_((x, y)), int(a_alt), m)

# 			attempts.append((c0, f"{val} front"))
# 			attempts.append((c1, f"{val} back"))
# 			attempts.append((c2, f"{val} front"))
# 			attempts.append((c3, f"{val} back"))

# 		ret.append(( (x, y), min(attempts)[1] ))
	
# 	return ret


# def _check_if_mask_matches(
# 	img: np.ndarray,
# 	mask: np.ndarray,
# 	x_y: tuple[float, float],
# 	a: float,
# 	m: np.ndarray
# ) -> float:
# 	h, w, _ = m.shape
# 	h, w = h // 2, w // 2
# 	x, y = x_y

# 	m_y, m_x, _ = img.shape

# 	im = img.copy()
# 	im = cv2.GaussianBlur(im, (GAUSS_K_SIZE, GAUSS_K_SIZE), 0)

# 	if x < w:
# 		x = w
# 	if x + w > m_x:
# 		x = m_x - w
# 	if y < h:
# 		y = h
# 	if y + h > m_y:
# 		y = m_y - h
	
# 	diff = np.abs(np.int16(im[y-h : y+h, x-w : x+w]) - np.int16(m[:h*2, :w*2]))
# 	diff[mask[y-h : y+h, x-w : x+w] == 0, :] = 0
# 	diff = np.mean(diff, axis=2, dtype=np.uint8)
# 	diff = np.uint8(diff)

# 	maxx = diff.max()
# 	diff[diff > maxx * (1 - AVG_ELEMENT_DROP_PTC)] = 0

# 	# cv2.imshow(f"Attempt {x_y} {a} {time.time()}", diff)
# 	return diff.mean()


