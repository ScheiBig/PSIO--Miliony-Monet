import atexit
import cv2
import numpy as np
import skimage as ski

import stage
import stage.tracking
from utils import ansi

REMOVE_BG = True

print(ansi.clear.ALL, end=None)
wait_time = 100

cap = cv2.VideoCapture("input/notes_on_belt.mkv")
keep = True

calibrating = True
card_was_on_screen = False
card_on_screen = False

bg_threshold = 20

calibration_size: list[float] = list()
calibration_points: list[tuple[float, float]] = list()

size_px_per_cm: float = None
speed_px_per_frame: float = None

atexit.register(lambda: stage.tracking.write_log("LOG"))

while keep:
	frame_was_read, frame = cap.read()
	frame_no = cap.get(cv2.CAP_PROP_POS_FRAMES)
	if not frame_was_read:
		# No next frame is present - video is finished
		cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 1)
		keep = False
		cv2.destroyAllWindows()
		cap.release()
		break

	frame_gr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	mask = stage.processing.threshold_mask(frame_gr, bg_threshold)

	if REMOVE_BG:
		bg_mask_sq = cv2.dilate(mask, ski.morphology.square(32))
		bg_mask_cr = cv2.dilate(mask, ski.morphology.disk(16))
		frame[np.logical_or(bg_mask_sq == 0, bg_mask_cr == 0)] = 255

	if calibrating:
		find_card = stage.calibration.find_card(mask)
		card_contours, card_rect, card_on_screen = find_card

		img_card = cv2.drawContours(frame.copy(), card_contours, -1, (0, 255, 255), 3)

		if card_was_on_screen and not card_on_screen:
			calibrating = False
			size_px_per_cm, speed_px_per_frame = stage.calibration.calculate_size_and_speed(
				calibration_size,
				calibration_points
			)
			# stage.detection.calibrated_size = stage.calibration.calibrated_size

		if len(card_rect) == 3:
			(r_x, r_y), (r_w, r_h), _ = card_rect
			calibration_size.append(r_w)
			calibration_size.append(r_h)
			calibration_points.append((r_x, r_y))

		card_was_on_screen = card_on_screen
		
		cv2.imshow("Detected shapes", img_card)
	else:
		rects, errors, shapes = stage.detection.detect_silhouettes(mask)
		rect_cont = [ np.int_(cv2.boxPoints(r)) for r in rects ]

		img_card = frame.copy()
		img_card = cv2.drawContours(img_card, rect_cont, -1, (0, 255, 0), 3)
		img_card = cv2.drawContours(img_card, errors, -1, (0, 0, 255), 3)

		img_card = cv2.drawContours(img_card, shapes, -1, (255, 0, 0), 1)
		rects_a, errors_a = stage.detection.detect_split_shapes(frame, shapes)

		rect_cont_a = [np.int_(cv2.boxPoints(r)) for r in rects_a]
		img_card = cv2.drawContours(img_card, rect_cont_a, -1, (127, 255, 127), 3)
		img_card = cv2.drawContours(img_card, errors_a, -1, (127, 127, 255), 3)


		# shape_and_hulls = [
		# 	(sh, cv2.convexHull(sh), cv2.convexHull(sh, returnPoints=False))
		# 		for sh in shapes
		# ]
		# shape_hulls = [ shh[1] for shh in shape_and_hulls ]
		# shape_defects = []
		# for shh in shape_and_hulls:
		# 	defs = cv2.convexityDefects(shh[0], shh[2])
		# 	if defs is not None:
		# 		defs = defs[:, 0, 2]
		# 	else:
		# 		continue
		# 	points = []
		# 	for i in defs:
		# 		points.append(shh[0][i])
		# 	shape_defects.append(np.array(points))
			
		

		# img_card = cv2.drawContours(img_card, shape_hulls, -1, (255, 255, 0), 3)
		# img_card = cv2.drawContours(img_card, shape_defects, -1, (255, 255, 255), 3)

		cv2.imshow("Detected shapes", img_card)

		objects: list[tuple[np.ndarray, float, stage.tracking.label]] = []

		for r in (rects + rects_a):
			objects.append((r[0], 1.0, stage.tracking.label.banknote_unknown))

		for e in (errors + errors_a):
			M = cv2.moments(e)
			cx = int(M['m10']/M['m00'])
			cy = int(M['m01']/M['m00'])
			objects.append((np.array([cx, cy]), 1.0, stage.tracking.label.unknown_shape))


		stage.tracking.track_objects(objects, frame_no)

	cv2.imshow("Mask", mask)
	cv2.imshow("Capture", frame)
	print(f"{ansi.cur.TOP_BEGIN}Numer klatki: {frame_no:10.0f}")
	print(f"Czas oczekiwania na klatkę: {wait_time:10.0f}")
	if size_px_per_cm is not None:
		print(f"Liczba pikseli na centymetr: {size_px_per_cm:10.4f}")
		print(f"Prędkość pikseli na klatkę: {speed_px_per_frame:10.4f}")

	match cv2.waitKey(wait_time):
		case 27:
			# Kill program if [ESC] is pressed
			keep = False
			cv2.destroyAllWindows()
			cap.release()
			print(ansi.clear.ALL)
			break
		case 111: # ord("o")
			wait_time = max(0, wait_time - 1)
		case 112: # ord("p")
			wait_time = min(200, wait_time + 1)
		case 117: # ord("u")
			wait_time = 0
		case 116: # ord("t")
			wait_time = 0
			cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 2)
		case 105: # ord("i")
			wait_time = 50
		case 113: # ord("q")
			cap.set(cv2.CAP_PROP_POS_FRAMES, int(input("Enter frame number to jump to:")))
