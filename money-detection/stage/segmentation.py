from typing import Literal, Optional, TypedDict, Union
import cv2
import numpy as np
from stage import calibration

POLYGON_APPROX_EPSILON = 8 # 0.01 * contour length of biggest banknote
POLYGON_TO_MIN_AREA_RECT_EPSILON = 0.04
SIDE_RATIO_MIN_AREA_RECT_EPSILON = 0.1
BANKNOTE_SIDE_RATIO = 2.0
BANKNOTE_SIDE_EPSILON_RATIO = 0.1
MIN_OBJECT_SIZE = 1000

def detect_silhouettes(img: np.ndarray) -> tuple[
	list[cv2.typing.RotatedRect], list[np.ndarray], list[np.ndarray]
]:
	'''
	Detects contours on mask image.

	This is first step of detection - it retrieves 4-sided elements and tests them for being
	possible banknote:
	* they must have same width-to-height ratio as banknotes,
	* they must have length of sides in range of available banknote side lengths (with some
	  small margin)
	* they must have minAreaRect with almost the same area.
	Other 4-sided elements, as well as 3-sided ones are being marked ar erroneous.

	Elements with more than 4 sides are marked as unknown - they require further investigation.

	Finally, all 4-and-less-sided elements that are touching sides of capture (they might be
	partially outside of the screen) are ignored - unknown elements are not, as they require
	further investigation.

	:param img: Black-and-white after-threshold mask image, that contains only white elements
	on black background.
	:return: Tuple of 3 lists: Valid rectangles, erroneous elements and unknown elements.
	'''

	global size_cache
	if size_cache is None:
		size_cache = {
			"min_w": calibration.calibrated_size["10"][0] * (1 - BANKNOTE_SIDE_EPSILON_RATIO),
			"max_w": calibration.calibrated_size["500"][0] * (1 + BANKNOTE_SIDE_EPSILON_RATIO),
			"min_h": calibration.calibrated_size["10"][1] * (1 - BANKNOTE_SIDE_EPSILON_RATIO),
			"max_h": calibration.calibrated_size["500"][1] * (1 + BANKNOTE_SIDE_EPSILON_RATIO),
		}

	min_y = 4
	max_y = img.shape[0] - 4

	contours, _ = cv2.findContours(
		img,
		cv2.RETR_TREE,
		cv2.CHAIN_APPROX_SIMPLE
	)

	if len(contours) == 0:
		return [], [], []

	result_rects: list[cv2.typing.RotatedRect] = []
	error_contours: list[np.ndarray] = []
	unknown_contours: list[np.ndarray]  = []

	for i, contour in enumerate(contours):
		poly_len = cv2.arcLength(contour, True)
		# poly_cont = cv2.approxPolyDP(
		# 	contour,
		# 	POLYGON_APPROX_EPSILON * poly_len,
		# 	True
		# )
		poly_cont = cv2.approxPolyDP(
			contour,
			POLYGON_APPROX_EPSILON,
			True
		)

		if len(poly_cont) == 4:

			match _test_shape_validity(poly_cont, (min_y, max_y)):
				case ("valid", rect):
					result_rects.append(rect)
				case "error":
					error_contours.append(poly_cont)
				case "ignored":
					continue

		elif len(poly_cont) == 3:
			if _test_shape_touches_side(contour, (min_y, max_y)):
				continue
			error_contours.append(poly_cont)
		else:
			if cv2.contourArea(poly_cont) > size_cache["min_w"] * size_cache["min_h"]:
				unknown_contours.append(poly_cont)
			elif cv2.contourArea(poly_cont) < MIN_OBJECT_SIZE:
				continue
			else:
				if _test_shape_touches_side(contour, (min_y, max_y)):
					continue
				error_contours.append(poly_cont)


	return result_rects, error_contours, unknown_contours


def detect_split_shapes(img: np.ndarray, unknown_shapes: list[np.ndarray]) -> tuple[
	list[cv2.typing.RotatedRect], list[np.ndarray]
]:
	'''
	Perform splitting of complex shapes, if it is possible (only supported for two banknotes).

	It is possible only to split shapes, that are not convex, as splitting relies on convexity
	defects to find split points - any shape that is already convex, is automatically assumed
	as erroneous.

	Splitting for shapes with two defects, tries to create two rectangles by splitting shape on
	those points - result should be two shapes with 6 vertices, which are further approximated into
	polygons (removes points lying slightly outside of banknotes). For those shapes, RotatedRect
	are created, which are tested same way as in ``detect_silhouettes`` function. It is possible
	for only one of split shapes to be valid.

	Splitting shapes with only one defect, relies on extrapolation - two neighboring points
	(of defect) are selected, and imaginary lines are drawn, which should project two new
	additional split points (theoretical) on sides of original shape. After that, both pairs
	defect-new point are tested in same way as described above - one that produces best results
	is selected, or shape is marked as erroneous if split is not possible.

	Additionally, any convex shapes and potentially erroneous ones, are discarded, if they are
	touching sides of capture (they might be partially outside of the screen).

	:param img: Black-and-white after-threshold mask image, that contains only white elements
	on black background - same one as passed to function ``detect_silhouettes``.
	:param unknown_shapes: shapes for further investigation returned from function
	``detect_silhouettes``.
	:return: Tuple of 2 lists: Valid rectangles and erroneous elements.
	'''

	min_y = 4
	max_y = img.shape[0] - 4

	if len(unknown_shapes) == 0:
		return [], []

	result_rects: list[cv2.typing.RotatedRect] = []
	error_contours: list[np.ndarray] = []

	for shape in unknown_shapes:
		i_hull = cv2.convexHull(shape, returnPoints= False)
		hull: np.ndarray = shape[i_hull[:, 0]]

		if np.all(np.isin(hull, shape, assume_unique=True)) and hull.shape == shape.shape:
			if _test_shape_touches_side(shape, (min_y, max_y)):
				continue
			error_contours.append(shape)
			continue
		
		defects = cv2.convexityDefects(shape, i_hull)
		i_defect_points = defects[:, 0, 2]
		i_defect_points.sort()

		if len(i_defect_points) == 2:
			
			splits = np.split(shape, i_defect_points)
			first: np.ndarray
			second: np.ndarray
			if len(splits) == 2:
				first, second = splits
			else:
				first = np.vstack([splits[0], splits[1][:1], splits[2]])
				second = np.vstack([splits[1], splits[2][:1]])

			res: list[cv2.typing.RotatedRect] = []
			err: list[np.ndarray] = []

			if first.size > 2:
				_fit_shape_or_approx_to_rect(first, res, err, (min_y, max_y))
			if second.size > 2:
				_fit_shape_or_approx_to_rect(second, res, err, (min_y, max_y))

			im1 = img.copy()

			cv2.drawContours(im1, [first, second], -1, (255, 255, 255), 2)
			cv2.imshow("Cnt", im1)

			result_rects.extend(res)
			error_contours.extend(err)

		elif len(i_defect_points) == 1:
			
			shape_i = np.hstack((
				shape[:, 0],
				np.arange(len(shape))[:, np.newaxis]
			))

			sh: np.ndarray = np.vstack((shape_i, shape_i, shape_i))
			sp = i_defect_points[0]
			dp: np.ndarray = sp + len(shape_i)

			dp_neighbors: np.ndarray = sh[dp - 1 : dp + 2]
			others: np.ndarray = sh[dp - len(shape_i) + 2 : dp - 1]

			new_split_points: list[tuple[np.ndarray, tuple[int, int, int]]] = []

			for i in range(len(others) - 1):
				i1 = _get_intersection((others[i], others[i+1]), dp_neighbors[:2])
				i2 = _get_intersection((others[i], others[i+1]), dp_neighbors[1:])

				x_min = min(others[i, 0], others[i+1, 0])
				x_max = max(others[i, 0], others[i+1, 0])
				y_min = min(others[i, 1], others[i+1, 1])
				y_max = max(others[i, 1], others[i+1, 1])

				if x_min < i1[0] \
					and x_max > i1[0] \
					and y_min < i1[1] \
					and y_max > i1[1]:
					new_split_points.append([i1, (others[i, 2], others[i+1, 2])])

				if x_min < i2[0] \
					and x_max > i2[0] \
					and y_min < i2[1] \
					and y_max > i2[1]:
					new_split_points.append([i2, (others[i, 2], others[i+1, 2])])

			shape_ex = np.vstack([shape, shape])

			# first try
			res: list[cv2.typing.RotatedRect] = []
			err: list[np.ndarray] = []

			if len(new_split_points) != 2:
				if _test_shape_touches_side(shape, (min_y, max_y)):
					continue
				error_contours.append(shape)
				continue
				

			p, (o0, o1) = new_split_points[0]

			o_min = min(o0, o1)
			o_max = max(o0, o1)
			sp_min = sp
			sp_max = sp

			if o_min < sp_min:
				o_min += len(shape)
			
			if o_max > sp_max:
				sp_max += len(shape)

			first = np.vstack([
				np.array([[p]]),
				shape_ex[sp_min : o_min + 1][:]
			])

			second = np.vstack([
				np.array([[p]]),
				shape_ex[o_max : sp_max + 1]
			])

			if first.size > 2:
				_fit_shape_or_approx_to_rect(first, res, err, (min_y, max_y))
			if second.size > 2:
				_fit_shape_or_approx_to_rect(second, res, err, (min_y, max_y))

			# second try
			res_alt: list[cv2.typing.RotatedRect] = []
			err_alt: list[np.ndarray] = []

			p, (o0, o1) = new_split_points[1]

			o_min = min(o0, o1)
			o_max = max(o0, o1)
			sp_min = sp
			sp_max = sp

			if o_min < sp_min:
				o_min += len(shape)
			
			if o_max > sp_max:
				sp_max += len(shape)

			first = np.vstack([
				np.array([[p]]),
				shape_ex[sp_min : o_min + 1][:]
			])

			second = np.vstack([
				np.array([[p]]),
				shape_ex[o_max : sp_max + 1]
			])

			if first.size > 2:
				_fit_shape_or_approx_to_rect(first, res_alt, err_alt, (min_y, max_y))
			if second.size > 2:
				_fit_shape_or_approx_to_rect(second, res_alt, err_alt, (min_y, max_y))

			if len(res) == len(res_alt) == 0:
				error_contours.append(shape)
			elif len(res) > len(res_alt):
				result_rects.extend(res)
				error_contours.extend(err)
			else:
				result_rects.extend(res_alt)
				error_contours.extend(err_alt)

		else:
			error_contours.append(shape)

	return result_rects, error_contours

def _test_shape_validity(
	contour: np.ndarray,
	min_max_y: tuple[float, float],
	test_area: bool= True
) -> Union[
	tuple[Literal["valid"], cv2.typing.RotatedRect],
	Literal["error", "ignored"]
] :
	'''
	Tests whether contour can describe valid banknote, with rules:
	* they must have same width-to-height ratio as banknotes,
	* they must have length of sides in range of available banknote side lengths (with some
	  small margin)
	* they must have minAreaRect with almost the same area.

	:param contour: Contour (best if approximated) of shape to test.
	:param min_max_y: Tuple with capture horizontal bounds ``(min_y, max_y)``.
	:return: ``("valid", rect)`` with ``rect`` being RotatedRect around shape, if test is passed,
	``"ignored"`` if test passed but ``rect`` would lay outside of bounds, or shape already touches
	bounds, ``"error"`` if test failed.
	'''

	if _test_shape_touches_side(contour, min_max_y):
		return "ignored"

	rect = cv2.minAreaRect(contour)
	_, (r_w, r_h), _ = rect
	w = max(r_w, r_h)
	h = min(r_w, r_h)
	side_ratio = h / w * BANKNOTE_SIDE_RATIO

	rect_area = (r_w * r_h)
	contour_area = cv2.contourArea(contour)

	box = np.int_(cv2.boxPoints(rect))

	if side_ratio < (1 + SIDE_RATIO_MIN_AREA_RECT_EPSILON) \
		and side_ratio > (1 - SIDE_RATIO_MIN_AREA_RECT_EPSILON) \
		and w > size_cache["min_w"] \
		and w < size_cache["max_w"] \
		and h > size_cache["min_h"] \
		and h < size_cache["max_h"] \
		and (not test_area or (rect_area / contour_area) < (1 + POLYGON_TO_MIN_AREA_RECT_EPSILON)):
		# (not x or y) <=> x implies y

		if _test_shape_touches_side(box, min_max_y):
			return "ignored"
		return "valid", rect
	else:
		return "error"


def _test_shape_touches_side(contour_or_box: np.ndarray, min_max_y) -> np.bool:
	'''
	Test whether shape touches sides of capture.

	:param contour_or_box: contour from ``cv2.findContour`` or any other function returning contour
	array, or box from ``cv2.boxPoints`` or any other function returning box around shape.
	:param min_max_y: tuple of minimal and maximal value to test shape to.
	:return: False, if all points in shape reside between min and max y.
	'''

	if len(contour_or_box.shape) == 2:
		return np.any(contour_or_box[:, 1] >= min_max_y[1]) \
			or np.any(contour_or_box[:, 1] <= min_max_y[0])
	else:
		return np.any(contour_or_box[:, 0, 1] >= min_max_y[1]) \
			or np.any(contour_or_box[:, 0, 1] <= min_max_y[0])
	

def _fit_shape_or_approx_to_rect(
	shape: np.ndarray,
	result_rects: list[cv2.typing.RotatedRect],
	error_contours: list[np.ndarray],
	min_max_y: tuple[float, float]
):
	'''
	Attempts to fit shape approximation as banknote rectangle, or shape itself
	
	:param shape: shape contour to test.
	:param result_rect: container for valid rectangles
	:param error_contours: container for error shapes
	'''

	# approx = cv2.approxPolyDP(
	# 	shape,
	# 	POLYGON_APPROX_EPSILON * cv2.arcLength(shape, True),
	# 	True
	# )
	approx = cv2.approxPolyDP(
		shape,
		POLYGON_APPROX_EPSILON,
		True
	)
	
	match _test_shape_validity(approx, min_max_y, False):
		case ("valid", rect):
			result_rects.append(rect)
		case "error":
			match _test_shape_validity(shape, min_max_y, False):
				case ("valid", rect):
					result_rects.append(rect)
				case "error":
					error_contours.append(shape)
				case "ignored":
					pass
		case "ignored":
			pass


def _get_intersection(
	line_1: tuple[np.ndarray, np.ndarray],
	line_2: tuple[np.ndarray, np.ndarray]
) -> Optional[np.ndarray]:
	'''
	Finds intersection of two lines, that are going through points.

	:param line_1: tuple of two 2d element that line goes through.
	:param line_2: tuple of two 2d element that line goes through.
	:return: Intersection point, or ``None`` if lines are parallel.
	'''

	x = 0
	y = 1

	p1a, p1b = line_1
	p2a, p2b = line_2

	# parallel, vertical
	if p1a[x] == p1b[x] and p2a[x] == p2b[x]:
		return

	# First vertical
	if p1a[x] == p1b[x]:
		m2: float = (p2b[y] - p2a[y]) / (p2b[x] - p2a[x])
		b2: float = p2a[y] - m2 * p2a[x]

		x_i = p1a[x]
		y_i = m2 * p1a[x] + b2

		return np.array((x_i, y_i), dtype= np.int_)

	# Second vertical
	if p2a[x] == p2b[x]:
		m1: float = (p1b[y] - p1a[y]) / (p1b[x] - p1a[x])
		b1: float = p1a[y] - m1 * p1a[x]

		x_i = p2a[x]
		y_i = m1 * p2a[x] + b1

		return np.array((x_i, y_i), dtype= np.int_)

	m1: float = (p1b[y] - p1a[y]) / (p1b[x] - p1a[x])
	m2: float = (p2b[y] - p2a[y]) / (p2b[x] - p2a[x])

	# parallel, slant or horizontal
	if m1 == m2:
		return

	b1: float = p1a[y] - m1 * p1a[x]
	b2: float = p2a[y] - m2 * p2a[x]

	x_i = (b2 - b1) / (m1 - m2)
	y_i = m1 * x_i + b1

	return np.array((x_i, y_i), dtype= np.int_)


sc = TypedDict("sc", {
	"min_w": float,
	"max_w": float,
	"min_h": float,
	"max_h": float,
})

size_cache: sc = None
