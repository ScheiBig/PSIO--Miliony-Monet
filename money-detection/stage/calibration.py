import math
import cv2
import numpy as np

from typing import TypedDict, Literal

CALIBRATION_CARD_SIDE_CM = 10.0

def find_card(img: np.ndarray) -> tuple[list[np.ndarray], cv2.typing.RotatedRect|tuple[()], bool]:
	'''
	Finds calibration card on image, assuming that card is only object currently on screen.

	:param img: Black-and-white after-threshold mask image, that contains only white card
	on black background.
	:return: if card is currently off-screen, then structure ``([], (), False)`` is returned,
	otherwise ``([box], rect, True)`` where ``box`` is contour of card, and ``rect`` is min-area
	``RotatedRect``.
	'''

	contours, _ = cv2.findContours(
		img,
		cv2.RETR_TREE,
		cv2.CHAIN_APPROX_SIMPLE
	)

	assert len(contours) <= 1, \
		"Function must be used only with calibration card on screen"

	if len(contours) == 0:
		return [], (), False

	contour = max(contours, key=cv2.contourArea)

	rect = cv2.minAreaRect(contour)
	_, (r_w, r_h), _ = rect

	if abs(r_w - r_h) < 4:
		box: np.ndarray = cv2.boxPoints(rect)
		box = box.astype(np.int_)

		return [box], rect, True

	return [], (), False


def calculate_size_and_speed(
		calibration_size: list[float],
		calibration_points: list[tuple[float, float]]
	) -> tuple[float, float]:
	'''
	Takes list of card side-lengths and card positions, returns size of centimeter
	in pixels and speed of belt movement in pixels per frame.

	Additionally, it initializes calibrated banknote sizes.

	:param calibration_size: List of side lengths of card - card is square, so heights
	and widths can be passed together.
	:param calibration_points: List of card positions (when fully on screen).
	:return: (size_px_per_cm, speed_px_per_frame).
	'''

	pair_n = len(calibration_points) - 1
	dists: list[float] = list()

	for i in range(pair_n):
		dists.append(_euclidean_distance(calibration_points[i], calibration_points[i+1]))

	px_size = sum(calibration_size) / len(calibration_size) / 10

	global calibrated_size
	calibrated_size = {
		k: (
			v_w / CALIBRATION_CARD_SIDE_CM * px_size,
			v_h / CALIBRATION_CARD_SIDE_CM * px_size
		)
			for k, (v_w, v_h) in _og_size.items()
	} # type: ignore [assignment] # this is partial dictionary construction
	calibrated_size["0"] = px_size

	global calibrated_speed
	calibrated_speed = sum(dists) / len(dists)

	return px_size, calibrated_speed


def _euclidean_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
	x1, y1 = p1
	x2, y2 = p2
	return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

Deno = Literal["10","20","50","100","200","500"]
deno_list = ["10","20","50","100","200","500"]

Pln = TypedDict("Pln", {
	"0": float,
	"10": tuple[float, float],
	"20": tuple[float, float],
	"50": tuple[float, float],
	"100": tuple[float, float],
	"200": tuple[float, float],
	"500": tuple[float, float],
})

_og_size: dict[Deno, tuple[int, int]] = {
	"10": (120, 60),
	"20": (126, 63),
	"50": (132, 66),
	"100": (138, 69),
	"200": (144, 72),
	"500": (150, 75),
}
'''
Original banknote  sizes, in millimeters
'''

calibrated_size: Pln = {} # type: ignore [typeddict-item] # late-init by calculate_size_and_speed(..)
'''
Sizes of banknotes, calibrated do pixels on capture stream. 
"0" key contains size of centimeter on frame in px.
'''

calibrated_speed: float
