from enum import StrEnum
import math

import numpy as np
from stage import calibration

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

log: list[log_entry] = []
currently_tracked_objects: list[tuple[np.ndarray, int, list[float], label]] = []

def track_objects(objects: list[tuple[np.ndarray, float, label]], frame_no: int):

	global currently_tracked_objects

	tracked_objects = [
		(p + np.array([0, calibration.calibrated_speed]), a, c, l)
		for p, a, c, l in currently_tracked_objects
	]

	found: list[tuple[np.ndarray, int, list[float]]] = []
	discovered: list[tuple[np.ndarray, int, list[float]]] = []

	while True:
		
		if len(objects) == 0 or len(tracked_objects) == 0:
			break
		el = tracked_objects.pop(0)

		objects.sort(key= lambda p: math.hypot(el[0][0] - p[0][0], el[0][1] - p[0][1]))
		near_el = objects.pop(0)

		found.append((near_el[0], el[1], el[2] + [near_el[1]], el[3]))

	if len(objects) > 0:

		discovered.extend([
			(p, frame_no, [c], l)
			for p, c, l in objects
		])

	if len(tracked_objects) > 0:

		for _, a, c, l in tracked_objects:
			log.append(log_entry(
				a, frame_no - 1, sum(c) / len(c), l
			))

	currently_tracked_objects = found + discovered

def write_log(path: str):
	global log

	log.extend([
		log_entry(a, -1, sum(c) / len(c), l)
		for _, a, c, l in currently_tracked_objects
	])

	with open(path, "w", encoding="utf-8") as f:

		for entry in log:
			f.write(str(entry) + "\n")
	
	log.clear()

