import atexit
import cv2
import cv2.typing as cv2_t
import numpy as np
import skimage as ski

# import stage
# import stage.tracking
from stage import *
from utils import *

REMOVE_BG = True

print(ansi.clear.ALL, end=None)
wait_time = 1

cap = cv2.VideoCapture("input/notes_on_belt.mkv")
keep = True

calibrating = True
card_was_on_screen = False
card_on_screen = False

bg_threshold = 20

calibration_size: list[float] = list()
calibration_points: list[tuple[float, float]] = list()

size_px_per_cm: float | None = None
speed_px_per_frame: float | None = None

# checkerboard: np.ndarray = np.array([[255, 240], [240, 255]]) \
checkerboard: np.ndarray = np.array([[30, 45], [45, 30]]) \
	.repeat(20, axis= 0) \
	.repeat(20, axis= 1)
checkerboard = np.tile(checkerboard, (50, 50))

# atexit.register(lambda: stage.tracking.write_log("LOG"))
atexit.register(lambda: tracking.write_log("LOG"))

times = spoiling_queue(maxsize= 10)

writer = cv2.VideoWriter(
	"./money-detection/notes_on_belt.mkv",
	cv2.VideoWriter.fourcc(*"xvid"),
	fps= 10,
	frameSize= (1716, 858),
	isColor= True
)

while keep:
	sw = stopwatch()
	frame_was_read, frame = cap.read()
	frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
	if not frame_was_read:
		# No next frame is present - video is finished
		cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 1)
		keep = False
		cv2.destroyAllWindows()
		cap.release()
		break

	frame_gr: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# mask = stage.processing.threshold_mask(frame_gr, bg_threshold)
	mask: np.ndarray = processing.threshold_mask(frame_gr, bg_threshold)

	frame_nobg = frame.copy()
	frame_nobg[mask == 0] = 0
	cv2.imshow("nobg", frame_nobg)

	frame_og = frame.copy()

	fr_out = np.zeros((858,1716,3), dtype= np.uint8)
	if REMOVE_BG:
		bg_mask_sq: np.ndarray = cv2.dilate(mask, ski.morphology.square(32))
		bg_mask_cr: np.ndarray = cv2.dilate(mask, ski.morphology.disk(16))
		ch: np.ndarray = checkerboard[0:frame.shape[0], 0:frame.shape[1]]
		cond = np.logical_or(bg_mask_sq == 0, bg_mask_cr == 0)
		frame[cond, 0] = ch[cond]
		frame[cond, 1] = ch[cond]
		frame[cond, 2] = ch[cond]
	
	ff = frame.copy()

	if calibrating:
		# find_card = stage.calibration.find_card(mask)
		find_card = calibration.find_card(mask)
		card_contours, card_rect, card_on_screen = find_card

		img_card = cv2.drawContours(frame.copy(), card_contours, -1, (0, 255, 255), 3)

		if card_was_on_screen and not card_on_screen:
			calibrating = False
			# size_px_per_cm, speed_px_per_frame = stage.calibration.calculate_size_and_speed(
			# 	calibration_size,
			# 	calibration_points
			# )
			size_px_per_cm, speed_px_per_frame = calibration.calculate_size_and_speed(
				calibration_size,
				calibration_points
			) 
			# stage.segmentation.calibrated_size = stage.calibration.calibrated_size

		if len(card_rect) == 3:
			(r_x, r_y), (r_w, r_h), _ = card_rect
			calibration_size.append(r_w)
			calibration_size.append(r_h)
			calibration_points.append((r_x, r_y))

		card_was_on_screen = card_on_screen
		
		cv2.imshow("Detected shapes", img_card)
	else:
		# rects, errors, shapes = stage.segmentation.detect_silhouettes(mask)
		rects: list[cv2_t.RotatedRect]
		errors: list[np.ndarray]
		shapes: list[np.ndarray]
		rects, errors, shapes = segmentation.detect_silhouettes(mask)
		rect_cont: list[np.ndarray] = [ np.asarray(cv2.boxPoints(r), dtype= np.int_) for r in rects ]

		img_card = frame.copy()
		img_card = cv2.drawContours(img_card, rect_cont, -1, (0, 255, 0), 3)
		img_card = cv2.drawContours(img_card, errors, -1, (0, 0, 255), 3)

		img_card = cv2.drawContours(img_card, shapes, -1, (255, 0, 0), 1)
		# rects_a, errors_a = stage.segmentation.detect_split_shapes(frame, shapes)
		rects_a, errors_a = segmentation.detect_split_shapes(frame_nobg, shapes)

		rect_cont_a = [ np.asarray(cv2.boxPoints(r), dtype= np.int_) for r in rects_a ]
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


		# val = stage.detection.detect_banknotes(frame_og, mask, (rects + rects_a))
		notes, errs = detection.detect_banknotes(frame_nobg, mask, (rects + rects_a))
		for n in notes:
			n.draw(ff)
		cv2.drawContours(ff, errors, -1, (0, 0, 255), 3)
		cv2.drawContours(ff, errors_a, -1, (127, 127, 255), 3)

		cv2.imshow("detection", ff)

		# for x_y, v in val:
		# 	img_card = cv2.putText(
		# 		img_card,
		# 		v,
		# 		np.int_([x_y[0] - 50, x_y[1]]),
		# 		cv2.FONT_HERSHEY_SIMPLEX,
		# 		0.75,
		# 		(127, 255, 0),
		# 		1
		# 	)

		# objects: list[tuple[np.ndarray, float, stage.tracking.label]] = []
		objects: list[tracking.track_obj] = []

		for r in (rects + rects_a):
			# objects.append((r[0], 1.0, stage.tracking.label.banknote_unknown))
			objects.append(((r[0][0],r[0][1]), 1.0, tracking.label.banknote_unknown))

		for e in (errors + errors_a):
			M = cv2.moments(e)
			cx = int(M['m10']/M['m00'])
			cy = int(M['m01']/M['m00'])
			# objects.append((np.array([cx, cy]), 1.0, stage.tracking.label.unknown_shape))
			objects.append(((cx, cy), 1.0, tracking.label.unknown_shape))

		# stage.tracking.track_objects(objects, frame_no)
		tracking.track_objects(objects, frame_no)

		cv2.imshow("Detected shapes", img_card)

		with open("./rect", "a") as f:
			for r in (rects + rects_a):
				f.write(f"{r}\n")

	# cv2.imshow("Mask", mask)
	# cv2.imshow("Capture", frame)
	et = sw.stop()
	times.put(et)
	fps = 1000 / times.avg()
	print(f"{ansi.cur.TOP_BEGIN}")
	print(f"{ansi.cur.TOP_BEGIN}Numer klatki: {frame_no:10.0f}")
	print(f"Czas opóźnienia klatki: {wait_time:10.0f}ms")
	print(f"Czas obliczenia klatki: {et:.2f}ms")
	print(f"Średnie FPS: {fps:.2f}")
	if size_px_per_cm is not None:
		print(f"Liczba pikseli na centymetr: {size_px_per_cm:10.4f}")
		print(f"Prędkość pikseli na klatkę: {speed_px_per_frame:10.4f}")

	fr_out[:, :frame.shape[1], :] = frame_og
	fr_out[:, -frame.shape[1]:, :] = ff

	writer.write(fr_out)

	match cv2.waitKey(wait_time):
		case 27:
			# Kill program if [ESC] is pressed
			keep = False
			cv2.destroyAllWindows()
			cap.release()
			print(ansi.clear.ALL)
			break
		case 111: # == ord("o")
			wait_time = max(0, wait_time - 1)
		case 112: # == ord("p")
			wait_time = min(200, wait_time + 1)
		case 117: # == ord("u")
			wait_time = 0
		case 116: # == ord("t")
			wait_time = 0
			cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 2)
		case 105: # == ord("i")
			wait_time = 50
		case 113: # == ord("q")
			cap.set(cv2.CAP_PROP_POS_FRAMES, int(input("Enter frame number to jump to:")))

writer.release()
