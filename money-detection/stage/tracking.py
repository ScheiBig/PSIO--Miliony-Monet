from enum import StrEnum
import math

import numpy as np
from stage import calibration, detection
from typing import Literal, Self
import cv2.typing as cv2_t
import copy

class label(StrEnum):
	banknote_10 = "Banknot 10zł"
	banknote_20 = "Banknot 20zł"
	banknote_50 = "Banknot 50zł"
	banknote_100 = "Banknot 100zł"
	banknote_200 = "Banknot 200zł"
	banknote_500 = "Banknot 500zł"

	banknote_unknown = "Banknot ?"

	unknown_shape = "Obcy obiekt o nierozpoznanym kształcie"
	unknown_label = "Obcy obiekt lub banknot nieznanego nominału"

	@classmethod
	def of(cls, value: Literal["10", "20", "50", "100", "200", "500"]):
		match value:
			case "10": return label.banknote_10
			case "20": return label.banknote_20
			case "50": return label.banknote_50
			case "100": return label.banknote_100
			case "200": return label.banknote_200
			case "500": return label.banknote_500


class log_entry(object):

	def __init__(self,
		first_appearance: int,
		lost_sight: int,
		confidence: float,
		type_label: label,
	) -> None:
		self.first_appearance: int = first_appearance
		self.lost_sight: int = lost_sight
		self.confidence: float = confidence
		self.type_label: label = type_label

	def __str__(self) -> str:

		lost_sight = f"{self.lost_sight:4.0f}" if self.lost_sight > 0 else "~ostatnią~"

		return f"Objekt śledzony między klatkami {self.first_appearance:4.0f} " \
			f"oraz {lost_sight} - z pewnością {self.confidence * 100:4.1f}% " \
			f"dopasowano etykietę {self.type_label}"

point = tuple[int|float, int|float]
log_row = tuple[detection.trackable, int, list[float], label]
track_obj = tuple[detection.trackable, float, label]

log: list[log_entry] = []
currently_tracked_objects: list[log_row] = []

def query_objects(
	objects: list[cv2_t.RotatedRect]
) -> tuple[list[cv2_t.RotatedRect], list[detection.detected_banknote]]:
	
	objects = objects.copy()
	global currently_tracked_objects

	tracked_objects: list[tuple[log_row, point]] = [
		((t, a, c, l), (t.center()[0], t.center()[1] - calibration.calibrated_speed))
		for t, a, c, l in currently_tracked_objects
		if isinstance(t, detection.detected_banknote)
	]

	retracked_objects: list[tuple[log_row, point]] = []

	found: list[detection.detected_banknote] = []
	discovered: list[cv2_t.RotatedRect] = []

	while True:

		if len(objects) == 0 or len(tracked_objects) == 0:
			break
		row, pt = tracked_objects.pop(0)
		tr, fr, cons, lb = row

		objects.sort(key= lambda p: math.hypot(
			pt[0] - p[0][0],
			pt[1] - p[0][1],
		))
		if math.hypot(
			pt[0] - objects[0][0][0], 
			pt[1] - objects[0][0][1]
		) > calibration.calibrated_size["0"]:
			retracked_objects.append((row, pt))
			continue
		near_el: cv2_t.RotatedRect = objects.pop(0)

		off_x = int(near_el[0][0] - tr.center()[0])
		off_y = int(near_el[0][1] - tr.center()[1])

		tr = copy.deepcopy(tr)

		match tr:
			case detection.banknote_front():
				tr.rect = ((tr.rect[0][0] + off_x, tr.rect[0][1] + off_y), tr.rect[1], tr.rect[2])
				if tr.bank_name is not None:
					tr.bank_name = (
						np.array(tr.bank_name[0]) + np.array([off_x, off_y]),
						tr.bank_name[1]
					)
				if tr.symbol is not None:
					tr.symbol = np.array(tr.symbol) + np.array([off_x, off_y])
				tr.denomination = np.array(tr.denomination) + np.array([off_x, off_y])
			case detection.banknote_back():
				tr.rect = ((tr.rect[0][0] + off_x, tr.rect[0][1] + off_y), tr.rect[1], tr.rect[2])
				if tr.bank_name is not None:
					tr.bank_name = (
						np.array(tr.bank_name[0]) + np.array([off_x, off_y]),
						tr.bank_name[1]
					)
				if tr.bank_symbol is not None:
					tr.bank_symbol = (
						np.array(tr.bank_symbol[0]) + np.array([off_x, off_y]),
						tr.bank_symbol[1]
					)
				tr.denomination = np.array(tr.denomination) + np.array([off_x, off_y])
				...
			case _: raise TypeError(f"Unknown type of detected banknote: {type(tr)}")

		found.append(tr)

	if len(objects) > 0:
		discovered.extend(objects)

	return (discovered, found)

def track_objects(objects: list[track_obj], frame_no: int) -> None:

	objects = objects.copy()
	global currently_tracked_objects

	tracked_objects: list[tuple[log_row, point]] = [
		((t, a, c, l), (t.center()[0], t.center()[1] - calibration.calibrated_speed))
		for t, a, c, l in currently_tracked_objects
	]
	
	retracked_objects: list[tuple[log_row, point]] = []

	found: list[log_row] = []
	discovered: list[log_row] = []

	while True:
		
		if len(objects) == 0 or len(tracked_objects) == 0:
			break
		el: tuple[log_row, point] = tracked_objects.pop(0)

		objects.sort(key= lambda p: math.hypot(
			el[1][0] - p[0].center()[0],
			el[1][1] - p[0].center()[1])
		)
		if math.hypot(
			el[1][0] - objects[0][0].center()[0], 
			el[1][1] - objects[0][0].center()[1]
		) > calibration.calibrated_size["0"]:
			retracked_objects.append(el)
			continue
		near_el: track_obj = objects.pop(0)

		found.append((near_el[0], el[0][1], el[0][2] + [near_el[1]], near_el[2]))

	if len(objects) > 0:
		discovered.extend([
			(p, frame_no, [c], l)
			for p, c, l in objects
		])

	tracked_objects.extend(retracked_objects)

	if len(tracked_objects) > 0:
		for (_, a, c, l), _ in tracked_objects:
			log.append(log_entry(
				a, frame_no - 1, sum(c) / len(c), l
			))

	currently_tracked_objects = found + discovered

def write_log(path: str) -> None:
	global log

	log.extend([
		log_entry(a, -1, sum(c) / len(c), l)
		for _, a, c, l in currently_tracked_objects
	])

	with open(path, "w", encoding="utf-8") as f:

		for entry in log:
			f.write(str(entry) + "\n")
	
	log.clear()

