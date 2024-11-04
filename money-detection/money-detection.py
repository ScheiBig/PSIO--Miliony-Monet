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
wait_time = 0

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
times_ex: list[list[float]] = []

# writer = cv2.VideoWriter(
# 	"./money-detection/notes_on_belt.mkv",
# 	cv2.VideoWriter.fourcc(*"xvid"),
# 	fps= 10,
# 	frameSize= (1716, 858),
# 	isColor= True
# )

while keep:
	tims: list[float] = []
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
	cv2.imshow("Original frame", frame)

	frame_gr: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	mask: np.ndarray = processing.threshold_mask(frame_gr, bg_threshold)

	frame_nobg = frame.copy()
	frame_nobg[mask == 0] = 0

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

		img_card = cv2.drawContours(frame.copy(), card_contours, -1, (255, 0, 255), 3)

		if card_was_on_screen and not card_on_screen:
			calibrating = False
			
			size_px_per_cm, speed_px_per_frame = calibration.calculate_size_and_speed(
				calibration_size,
				calibration_points
			) 

		if len(card_rect) == 3:
			(r_x, r_y), (r_w, r_h), _ = card_rect
			calibration_size.append(r_w)
			calibration_size.append(r_h)
			calibration_points.append((r_x, r_y))

		card_was_on_screen = card_on_screen
		
		cv2.imshow("Detected shapes", img_card)
	else:
		rects: list[cv2_t.RotatedRect]
		errors: list[np.ndarray]
		shapes: list[np.ndarray]

		## Simple segmentation ##
		s_w = stopwatch()
		rects, errors, shapes = segmentation.get_silhouettes(mask)
		tims.append(s_w.stop())

		## Complex segmentation ##
		s_w.start()
		rects_a, errors_a = segmentation.get_split_shapes(mask, shapes)
		tims.append(s_w.stop())

		## Join results ##
		rects.extend(rects_a)
		errors.extend(errors_a)

		## Query tracked objects ##
		s_w.start()
		new, known = tracking.query_objects(rects)
		tims.append(s_w.stop())
		
		## Run detection only on new objects ##
		s_w.start()
		notes, unkn = detection.detect_banknotes(frame_nobg, mask, new)
		tims.append(s_w.stop())
		
		## Translate errors for further usage ##
		f_errors = [ detection.erroneous_object(er) for er in errors ]

		## Add cached notes to newly detected ##
		notes.extend(known)

		## Draw all shapes ##
		for n in notes:
			n.draw(ff)
		for un in unkn:
			un.draw(ff)
		for er in f_errors:
			er.draw(ff)
		cv2.imshow("Detected shapes", ff)

		## Track all objects ##
		s_w.start()
		objects: list[tracking.track_obj] = []

		for n in notes:
			objects.append((n, n.confidence() / 100, tracking.label.of(n.value)))
		for un in unkn:
			objects.append((un, 0, tracking.label.unknown_label))
		for er in f_errors:
			objects.append((er, 1.0, tracking.label.unknown_shape))

		tracking.track_objects(objects, frame_no)
		tims.append(s_w.stop())

		tims.append(len(notes) + len(unkn) + len(f_errors))
		tims.append(len(unkn) + len(new))

	## Print debug info to terminal ##
	et = sw.stop()
	times.put(et)
	tims.insert(0, 1000 / et if et != 0.0 else float("nan"))
	tims.insert(0, et)
	times_ex.append(tims)
	fps = 1000 / times.avg()
	print(f"{ansi.cur.TOP_BEGIN}")
	print(f"{ansi.cur.TOP_BEGIN}Numer klatki: {frame_no:10.0f}")
	print(f"Czas opóźnienia klatki: {wait_time:10.0f}ms")
	print(f"Czas obliczenia klatki: {et:.2f}ms")
	print(f"Średnie FPS: {fps:.2f}")
	if size_px_per_cm is not None:
		print(f"Liczba pikseli na centymetr: {size_px_per_cm:10.4f}")
		print(f"Prędkość pikseli na klatkę: {speed_px_per_frame:10.4f}")

	# fr_out[:, :frame.shape[1], :] = frame_og
	# fr_out[:, -frame.shape[1]:, :] = ff

	# writer.write(fr_out)

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

# writer.release()
with open("TIMES.CSV", "w", encoding="utf-8") as tms:
	tms.write("all_ms;all_fps;segm_ms;split_ms;query_ms;detect_ms;track_ms;objects;unknowns\n")
	for t in times_ex:
		for i, v in enumerate(t):
			tms.write(f"{v:8.2f};")
		tms.write("\n")

