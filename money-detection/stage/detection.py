import math
import re
import time
from typing import Literal, TypedDict, cast
import cv2
import cv2.typing as cv2_t
import numpy as np
import numpy.ma as ma
from stage import calibration
from abc import abstractmethod, ABC
import easyocr # type: ignore [import-untyped] # kinda cringe, ngl
import skimage as ski
import difflib

AVG_ELEMENT_DROP_PTC = 0.05
AVG_VALID_AVG_MAX = 24
GAUSS_K_SIZE = 5

reader = easyocr.Reader(["en"])

int_point = tuple[int, int]
int_box = np.ndarray|tuple[int_point, int_point, int_point, int_point]
eOcr_res = tuple[
	int_box, 
	str, 
	float
]
'''
Describes type of singular result of ``easyocr.Reader#readtext`` with specified
option ``detail= 1`` (which is the default value). Result consists of list 
of tuples, each containing bounding box, string label and 0-1 value 
of confidence; bounding box is defined as list of four integer 2d points.

**This type describes singular result, typing result
of ``easyocr.Reader#readtext`` should be: ``list[eOcr_res]``**.
'''

num_trans_dict = {
	ord("o"): ord("0"),
	ord("O"): ord("0"),
	ord("D"): ord("0"),
	ord("Q"): ord("0"),
	ord("C"): ord("0"),
	ord("c"): ord("0"),

	ord("i"): ord("1"),
	ord("I"): ord("1"),
	ord("J"): ord("1"),
	ord("T"): ord("1"),
	ord("t"): ord("1"),
	ord("|"): ord("1"),
	ord("]"): ord("1"),
	ord("["): ord("1"),

	ord("Z"): ord("2"),
	ord("z"): ord("2"),

	ord("s"): ord("5"),
	ord("S"): ord("5"),
}
'''
Dictionary for translating 
'''

class trackable(ABC):

	@abstractmethod
	def draw(self, img: np.ndarray) -> np.ndarray:
		'''
		Draws this object features onto provided image.

		:param img: image to draw object to.
		:return: ``img`` after modifications.
		'''
		raise NotImplementedError()
	
	def center(self) -> int_point:
		'''
		Returns center of mass of this object.
		'''
		raise NotImplementedError()

class detected_banknote(trackable):

	def __init__(self,
		value: calibration.Deno,
		rect: cv2_t.RotatedRect,
		denomination: int_box,
		side: Literal["front", "back"],
		bank_name: tuple[int_box, float] | None = None
	) -> None:
		
		self.value: calibration.Deno = value
		self.rect: cv2_t.RotatedRect = rect
		self.denomination: int_box = denomination
		self.side: Literal["front", "back"] = side
		self.bank_name: tuple[int_box, float] | None = bank_name
		pass

	@abstractmethod
	def confidence(self) -> int:
		'''
		Calculates confidence of detection.

		:return: % of confidence that label was assigned correctly.
		'''
		raise NotImplementedError()
	
	def center(self) -> int_point:
		(c_x, c_y), _, _ = self.rect
		return (int(c_x), int(c_y))


class banknote_front(detected_banknote):

	def __init__(self,
		value: calibration.Deno,
		rect: cv2_t.RotatedRect,
		denomination: int_box,
		bank_name: tuple[int_box, float] | None = None,
		symbol: int_box | None = None,
	):
		super().__init__(value, rect, denomination, "front", bank_name)

		self.symbol: int_box | None = symbol

	def draw(self, img: np.ndarray) -> np.ndarray:
		b_x, b_y, _, _ = cv2.boundingRect(cv2.boxPoints(self.rect))

		cv2.drawContours(
			img,
			[cv2.boxPoints(self.rect).astype(np.int_)],
			-1,
			(0, 255, 0),
			3
		)
		cv2.putText(
			img,
			f"{self.value}zl :: p {self.confidence()}%",
			(b_x + 4, b_y + 4),
			cv2.FONT_HERSHEY_SIMPLEX,
			1,
			(0, 137, 0),
			2
		)
		# if self.bank_name is not None:
		# 	cv2.putText(
		# 		img,
		# 		f"NBP: {self.bank_name[1]}/18 liter",
		# 		(b_x + 4, b_y + 20),
		# 		cv2.FONT_HERSHEY_SIMPLEX,
		# 		0.75,
		# 		(64, 255, 64),
		# 		1
		# 	)

		cv2.drawContours(
			img,
			[np.array(self.denomination)],
			-1,
			(255, 255, 0),
			2
		)
		
		if self.bank_name is not None:
			cv2.drawContours(
				img,
				[np.array(self.bank_name[0])],
				-1,
				_confidence_color(self.bank_name[1]),
				2
			)
		if self.symbol is not None:
			cv2.drawContours(
				img,
				[np.array(self.symbol)],
				-1,
				(255, 255, 0),
				2
			)

		return img


	def confidence(self) -> int:
		conf = 0.6
		if self.symbol is not None:
			conf += 0.2
		if self.bank_name is not None:
			conf += self.bank_name[1] * 0.2

		return int(conf * 100)

class banknote_back(detected_banknote):

	def __init__(self,
		value: calibration.Deno,
		rect: cv2_t.RotatedRect,
		denomination: int_box,
		bank_name: tuple[int_box, float] | None = None,
		bank_symbol: tuple[int_box, float] | None = None,
	):
		super().__init__(value, rect, denomination, "back", bank_name)

		self.bank_symbol: tuple[int_box, float] | None = bank_symbol

	def draw(self, img: np.ndarray) -> np.ndarray:
		b_x, b_y, b_w, b_h = cv2.boundingRect(cv2.boxPoints(self.rect))

		cv2.drawContours(
			img,
			[cv2.boxPoints(self.rect).astype(np.int_)],
			-1,
			(0, 255, 0),
			3
		)
		cv2.putText(
			img,
			f"{self.value}zl :: p {self.confidence()}%",
			(b_x + 4, b_y - 4),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.75,
			(0, 137, 0),
			2
		)
		# if self.bank_name is not None:
		# 	cv2.putText(
		# 		img,
		# 		f"NBP: {self.bank_name[1]}/18 liter",
		# 		(b_x + 4, b_y + b_h + 20),
		# 		cv2.FONT_HERSHEY_SIMPLEX,
		# 		0.75,
		# 		(0, 137, 0),
		# 		2
		# 	)

		cv2.drawContours(
			img,
			[np.array(self.denomination, dtype= np.int_)],
			-1,
			(255, 255, 0),
			2
		)
		if self.bank_name is not None:
			cv2.drawContours(
				img,
				[np.array(self.bank_name[0])],
				-1,
				_confidence_color(self.bank_name[1]),
				2
			)
		if self.bank_symbol is not None:
			cv2.drawContours(
				img,
				[np.array(self.bank_symbol[0])],
				-1,
				_confidence_color(self.bank_symbol[1]),
				2
			)

		return img

	def confidence(self) -> int:
		conf = 0.6
		if self.bank_symbol is not None:
			conf += 0.2
		if self.bank_name is not None:
			conf += self.bank_name[1] * 0.2

		return int(conf * 100)


class undetected_object(trackable):

	def __init__(self,
		rect: cv2_t.RotatedRect,
	) -> None:
		self.rect: cv2_t.RotatedRect = rect
		
	def draw(self, img: np.ndarray) -> np.ndarray:
		cv2.drawContours(
			img,
			[cv2.boxPoints(self.rect).astype(np.int_)],
			-1,
			(0, 255, 255),
			2
		)
		return img
	
	def center(self) -> int_point:
		(c_x, c_y), _, _ = self.rect
		return (int(c_x), int(c_y))
	

class erroneous_object(trackable):
	
	def __init__(self,
		contour_or_rotatedRect: np.ndarray | cv2_t.RotatedRect
	) -> None:
		self.contour: np.ndarray
		if isinstance(contour_or_rotatedRect, np.ndarray):
			self.contour = contour_or_rotatedRect.astype(np.int_)
		else:
			self.contour = cv2.boxPoints(contour_or_rotatedRect).astype(np.int_)

	def draw(self, img: np.ndarray) -> np.ndarray:
		cv2.drawContours(
			img,
			[self.contour],
			-1,
			(64, 64, 255),
			2
		)
		return img
		
	def center(self) -> int_point:
		M: cv2_t.Moments = cv2.moments(self.contour)
		c_x = int(M["m10"]/M["m00"])
		c_y = int(M["m01"]/M["m00"])

		return (c_x, c_y)

def detect_banknotes(
	img: np.ndarray,
	mask: np.ndarray,
	rects: list[cv2_t.RotatedRect]
) -> tuple[list[detected_banknote], list[undetected_object]]:
	
	ret: tuple[list[detected_banknote], list[undetected_object]] = ([], [])

	for i, rect in enumerate(rects):
		(r_x, r_y), (r_w, r_h), r_a = rect
		r_x, r_y, r_w, r_h = int(r_x), int(r_y), int(r_w), int(r_h)
		bbox: cv2_t.Rect = cv2.boundingRect(cv2.boxPoints(rect).astype(dtype= np.int_))
		b_x, b_y, b_w, b_h = bbox
		
		if r_w < r_h:
			r_h, r_w = r_w, r_h
			r_a = (r_a + 90) % 360


		# create new image to paste and rotate banknote
		# side = int(math.hypot(r_w, r_h)) + 2
		side = math.ceil(math.hypot(r_w, r_h)) + 10
		banknote: np.ndarray = np.zeros((side, side, 3), dtype=np.uint8)
		
		# copy banknote ROI into center of new image
		off_x: int = max(side - b_w, 0) // 2
		off_y: int = max(side - b_h, 0) // 2
		banknote[off_y : off_y + b_h, off_x : off_x + b_w, :] = img[b_y : b_y + b_h, b_x : b_x + b_w, :]

		# rotate banknote into vertical position
		r_mat: np.ndarray = cv2.getRotationMatrix2D((side // 2, side // 2), r_a, 1)
		banknote = cv2.warpAffine(banknote, r_mat, (side, side)).astype(np.uint8)
		# cv2.imshow(f"{time.time()}", banknote)
		# remove outside bands
		off_x = (side - r_w) // 2
		off_y = (side - r_h) // 2
		banknote = banknote[off_y : off_y + r_h, off_x : off_x + r_w, :]
		flip_banknote = np.rot90(banknote, 2)

		# cv2.imshow(f"bn {i}", banknote)

		# try to detect in each orientation
		note: detected_banknote | None

		note = _detect_front(
			banknote,
			((r_x, r_y), (r_w, r_h), r_a),
			i,
		)
		if note is not None:
			ret[0].append(note)
			continue

		note = _detect_front(
			flip_banknote,
			((r_x, r_y), (r_w, r_h), (r_a + 180) % 360),
			i,
		)
		if note is not None:
			ret[0].append(note)
			continue

		note = _detect_back(
			banknote,
			((r_x, r_y), (r_w, r_h), r_a % 360),
			i,
		)
		if note is not None:
			ret[0].append(note)
			continue

		note = _detect_back(
			flip_banknote,
			((r_x, r_y), (r_w, r_h), (r_a + 180) % 360),
			i,
		)
		if note is not None:
			ret[0].append(note)
			continue

		ret[1].append(undetected_object(rect))

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

def _rotate_box(box: int_box, origin: tuple[int, int], angle: float):
	"""
	Rotate a box counterclockwise by a given angle around a given origin.
	origin
	:param point: box to rotate.
	:param origin: origin point of rotation.
	:param angle: angle of rotation, in degrees.
	:return: new position of point after rotation.
	"""
	angle = math.radians(angle)
	o_x, o_y = origin
	abox = np.array(box)
	p_x = abox[:,0]
	p_y = abox[:,1]
	q_x = o_x + math.cos(angle) * (p_x - o_x) - math.sin(angle) * (p_y - o_y)
	q_y = o_y + math.sin(angle) * (p_x - o_x) + math.cos(angle) * (p_y - o_y)
	return np.column_stack((q_x, q_y)).astype(np.int_)

rotatedIntRect = tuple[tuple[int, int], tuple[int, int], float]

def _detect_front(
	banknote: np.ndarray,
	og_rect: rotatedIntRect,
	i: int|None = None
) -> banknote_front | None:
	(r_x, r_y), (r_w, r_h), r_a = og_rect

	c = calibration.calibrated_size["0"]

	## Denomination ##

	# We will need to rotate the cutting
	W = int(c * 2.0)
	H = int(c * 3.8)
	den: np.ndarray = banknote[:H, :W, :]
	den = cv2.cvtColor(den, cv2.COLOR_BGR2GRAY)
	den = np.rot90(den, -1)

	thr: int = min(ski.filters.threshold_multiotsu(den[den != 0], 2))
	den = np.uint8(den < thr) * 255

	# cv2.imshow(f"den {i}", den) ## >debug<

	read_den: list[eOcr_res] = reader.readtext(den)
	den_val: eOcr_res
	d_box: int_box|None
	d_str: str|None
	d_conf: float|None
	for res in read_den:
		d_box, d_str, d_conf = res
		d_str = d_str.translate(num_trans_dict)[:3]
		if d_str in calibration.deno_list:
			d_box = np.flip(np.array(d_box), axis= 1)
			d_box[:, 1] = H - d_box[:, 1] 
			den_val = (d_box, d_str, d_conf)
			break
	else:
		l = next(iter(read_den), (None, "---"))
		# print(f"b_den_unk {i}", next(iter(read_den), (None, "---"))[1])
		return None
	
	den_box: np.ndarray = _place_bbox_back_to_image(
		den_val[0],
		1,
		None,
		og_rect
	)

	## Bank name ##

	if int(den_val[1]) <= 50:
		W = int(c * 3.8)
		H = int(c * 2.2)
		W_off = int(c * 2.6)
		H_off = int(c * 0.0)
	else:
		W = int(c * 3.5)
		H = int(c * 2.2)
		W_off = int(c * 4.2)
		H_off = int(c * 0.0)

	b_name: np.ndarray = banknote[H_off:(H_off + H), W_off:(W_off + W), :]
	b_name = cv2.cvtColor(b_name, cv2.COLOR_BGR2GRAY)
	thr = min(ski.filters.threshold_multiotsu(b_name[b_name != 0], 2))
	b_name = np.uint8(b_name < thr) * 255
	read_b_name: list[eOcr_res] = reader.readtext(b_name)
	n_box: int_box|None
	n_str: str|None
	n_conf: float|None
	if len(read_b_name) == 0:
		n_box, n_str, n_conf = (None, None, None)
	else:
		if len(read_b_name) > 1:
			n_box_all = np.array([r[0] for r in read_b_name])
			s_0, s_1, s_2 = n_box_all.shape
			n_box_all = n_box_all.reshape((s_0 * s_1, s_2))
			v_min = n_box_all.min(axis= 0)
			v_max = n_box_all.max(axis= 0)
			n_box = (
				(v_min[0], v_min[1]),
				(v_max[0], v_min[1]),
				(v_max[0], v_max[1]),
				(v_min[0], v_max[1]),
			)
			n_str = " ".join([r[1] for r in read_b_name])
		else:
			n_box, n_str, _ = read_b_name[0]

		n_str = re.sub(r'[^\w]', ' ', n_str)
		n_box = _place_bbox_back_to_image(n_box, 1, (W_off, H_off), og_rect)
		n_conf = difflib.SequenceMatcher(
			lambda ch: ch == " ",
			n_str.upper(),
			"NARODOWY BANK POLSKI"
		).ratio()

	## Note symbol ##

	W = int(c * 1.8)
	H = int(c * 1.8)
	if den_val[1] == "50":
		W = int(c * 2.4)

	b_symbol: np.ndarray = banknote[-H:, :W, :]
	b_symbol = cv2.cvtColor(b_symbol, cv2.COLOR_BGR2GRAY)
	thr = min(ski.filters.threshold_multiotsu(b_symbol[b_symbol != 0], 2)) - 5
	b_symbol = np.uint8(b_symbol < thr) * 255
	b_symbol = cv2.morphologyEx(b_symbol, cv2.MORPH_CLOSE, ski.morphology.disk(2))
	cv2.floodFill(b_symbol, None, (W//2, H//2), 255) # type: ignore [call-overload]

	# cv2.imshow(f"b_name {i}", b_symbol) ## >debug<

	## Results ##

	n_res: tuple[int_box, float] | None = None
	s_res: tuple[int_box, float] | None = None
	if n_box is not None \
			and n_conf is not None \
			and n_conf != 0 \
	:
		n_res = (n_box, n_conf)
	# if s_box is not None \
	# 		and s_conf is not None \
	# 		and s_conf != 0 \
	# :
	# 	s_res = (s_box, s_conf)

	return banknote_front(
		cast(calibration.Deno, den_val[1]),
		og_rect,
		den_box,
		n_res,
	)

def _detect_back(
	banknote: np.ndarray,
	og_rect: rotatedIntRect,
	i: int|None = None
) -> banknote_back | None:
	(r_x, r_y), (r_w, r_h), r_a = og_rect

	c = calibration.calibrated_size["0"]

	### Denomination ###

	W = int(c * 3.2)
	H = int(c * 3)
	den: np.ndarray = banknote[:H , -W:, :]
	den = cv2.cvtColor(den, cv2.COLOR_BGR2GRAY)

	## thr: int = min(ski.filters.threshold_multiotsu(den, 3)) - 10
	# Masking off removed background - if this was not done, then in some
	#   cutouts, there would be bands of "0"s, that would add chimney
	#   in histogram, which throws-off results of OTSU thresholding.
	thr: int = min(ski.filters.threshold_multiotsu(den[den != 0], 3)) - 10
	den = np.uint8(den < thr) * 255
	den = cv2.resize(
		den,
		(den.shape[1] * 2, den.shape[0] * 2),
		interpolation= cv2.INTER_LINEAR
	)
	den = cv2.morphologyEx(den, cv2.MORPH_OPEN, ski.morphology.disk(1))

	read_den: list[eOcr_res] = reader.readtext(den)
	den_val: eOcr_res
	d_box: int_box|None
	d_str: str|None
	d_conf: float|None
	for res in read_den:
		d_box, d_str, d_conf = res
		d_str = d_str.translate(num_trans_dict)
		if d_str in calibration.deno_list:
			den_val = (d_box, d_str, d_conf)
			break
	else:
		# l = next(iter(read_den), (None, "---"))
		# print(f"b_den_unk {i}", next(iter(read_den), (None, "---"))[1])
		return None
	
	den_box: np.ndarray = _place_bbox_back_to_image(
		den_val[0],
		2,
		(banknote.shape[1] - W, 0),
		og_rect
	)

	### Bank name ###

	W = int(c * 10)
	H = int(c * 1.5)

	b_name: np.ndarray = banknote[:H, :W, :]
	b_name = cv2.cvtColor(b_name, cv2.COLOR_BGR2GRAY)

	thr = min(ski.filters.threshold_multiotsu(b_name[b_name != 0], 3))
	b_name = np.uint8(b_name < thr) * 255
	read_b_name: list[eOcr_res] = reader.readtext(b_name)
	n_box: int_box|None
	n_str: str|None
	n_conf: float|None
	if len(read_b_name) == 0:
		n_box, n_str, n_conf = (None, None, None)
	else:
		n_box, n_str, _ = read_b_name[0]
		n_str = re.sub(r'[^\w]', ' ', n_str)

		n_box = _place_bbox_back_to_image(n_box, 1, None, og_rect)
		n_conf = difflib.SequenceMatcher(
			lambda ch: ch == " ",
			n_str.upper(),
			"NARODOWY BANK POLSKI"
		).ratio()

	## Bank symbol ##

	W = int(c * 3)
	H = int(c * 1.5)
	H_off = int(c * 0.3)

	b_symbol: np.ndarray = banknote[-(H+H_off):-H_off, -W:, :]
	b_symbol = cv2.cvtColor(b_symbol, cv2.COLOR_BGR2GRAY)

	thr = min(ski.filters.threshold_multiotsu(b_symbol[b_symbol != 0], 2)) - 5
	b_symbol = np.uint8(b_symbol < thr) * 255
	b_symbol = cv2.resize(
		b_symbol,
		(b_symbol.shape[1] * 2, b_symbol.shape[0] * 2),
		interpolation= cv2.INTER_LINEAR
	)
	read_b_symbol: list[eOcr_res] = reader.readtext(b_symbol)
	s_box: int_box|None
	s_str: str|None
	s_conf: float|None
	if len(read_b_symbol) == 0:
		s_box, s_str, s_conf = (None, None, None)
	else:
		s_box, s_str, _ = read_b_symbol[0]
		s_str = re.sub(r'[^\w]', ' ', s_str)

		s_box = _place_bbox_back_to_image(
			s_box,
			2,
			(banknote.shape[1] - W, banknote.shape[0] - H - H_off),
			og_rect,
		)
		s_conf = difflib.SequenceMatcher(
			lambda ch: ch == " ",
			s_str.upper(),
			"NBP"
		).ratio()

	## Results ##

	n_res: tuple[int_box, float] | None = None
	s_res: tuple[int_box, float] | None = None
	if n_box is not None \
			and n_conf is not None \
			and n_conf != 0 \
	:
		n_res = (n_box, n_conf)
	if s_box is not None \
			and s_conf is not None \
			and s_conf != 0 \
	:
		s_res = (s_box, s_conf)
	# if n_res is None and s_res is None:
	# 	return None

	return banknote_back(
		cast(calibration.Deno, den_val[1]),
		og_rect,
		den_box,
		n_res,
		s_res
	)

def _get_box_middle(box: int_box) -> int_point:
	return tuple(
		np.array(box)\
			.mean(axis= 0, dtype= np.int_)
	)

def _place_bbox_back_to_image(
	bbox: int_box,
	bbox_scale: int,
	offset: int_point|None,
	banknote_pos: cv2_t.RotatedRect,
) -> np.ndarray:
	(r_x, r_y), (r_w, r_h), r_a = banknote_pos
	den_box: np.ndarray = np.array(bbox, dtype= np.int_)
	# Return bounding box to original size
	den_box = den_box // bbox_scale
	# Offset bounding box - from cutout position
	den_box += np.array(offset if offset is not None else (0, 0), dtype= np.int_)
	# Offset bounding box on image, relative to banknote center, then
	#   relative to banknote size
	den_box += np.array((r_x - r_w//2, r_y - r_h//2), dtype= np.int_)
	# Rotate bounding box to original banknote rotation, pivoting on banknote
	#   center point
	den_box = _rotate_box(den_box, (int(r_x), int(r_y)), r_a)
	return den_box


def _confidence_color(conf: float) -> tuple[int, int, int]:
	r: float
	g: float
	# if conf > 0.5:
	# 	r = 180 - 180 * (conf * 2 - 1)
	# 	g = 150
	# else:
	# 	r = 255 - 75 * conf * 2
	# 	g = conf * 2 * 150
	if conf > 0.5:
		r = 255 - 255 * (conf * 2 - 1)
		g = 255
	else:
		r = 255
		g = 255 - 255 * conf * 2
	
	return (64, int(g), int(r))
