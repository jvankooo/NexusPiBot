import sys
import cv2
import time
import numpy as np
import pickle
import math


# Bot Controls
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)
 
Motor1a = 35    # Input Pin
Motor1b = 37    # Input Pin
Motor1e = 31	# Enable 1
Motor2a = 38    # Input Pin
Motor2b = 40    # Input Pin
Motor2e = 33	# Enable 2
Led1 = 32

GPIO.setup(Led1, GPIO.OUT) 
GPIO.setup(Motor1a,GPIO.OUT)
GPIO.setup(Motor1b,GPIO.OUT)
GPIO.setup(Motor1e,GPIO.OUT)
GPIO.setup(Motor2a,GPIO.OUT)
GPIO.setup(Motor2b,GPIO.OUT)
GPIO.setup(Motor2e,GPIO.OUT)

BotL = GPIO.PWM(Motor1e, 100)
BotR = GPIO.PWM(Motor2e, 100)
BotR.start(0)
BotL.start(0)

def ledon():
	print("BLINK")
	GPIO.output(Led1,GPIO.HIGH)
	
def ledoff():
	print("BLINK")
	GPIO.output(Led1,GPIO.LOW)

def botenable(val):
	print("")
	BotL.ChangeDutyCycle(val)
	BotR.ChangeDutyCycle(val)


def botforward():
	print("FORWARD")
	GPIO.output(Motor1a,GPIO.LOW)
	GPIO.output(Motor1b,GPIO.HIGH)
	GPIO.output(Motor2a,GPIO.LOW)
	GPIO.output(Motor2b,GPIO.HIGH)
	
def botbackward():
	print("BACKWARD")
	GPIO.output(Motor1a,GPIO.HIGH)
	GPIO.output(Motor1b,GPIO.LOW)
	GPIO.output(Motor2a,GPIO.HIGH)
	GPIO.output(Motor2b,GPIO.LOW)
	
def botright():
	print("RIGHT")
	GPIO.output(Motor1a,GPIO.HIGH)
	GPIO.output(Motor1b,GPIO.LOW)
	GPIO.output(Motor2a,GPIO.LOW)
	GPIO.output(Motor2b,GPIO.HIGH)
	
def botleft():
	print("LEFT")
	GPIO.output(Motor1a,GPIO.LOW)
	GPIO.output(Motor1b,GPIO.HIGH)
	GPIO.output(Motor2a,GPIO.HIGH)
	GPIO.output(Motor2b,GPIO.LOW)
	
def botstop():
	print("STOP")
	BotR.ChangeDutyCycle(0)
	BotL.ChangeDutyCycle(0)

def blink_led():
	ledon()
	time.sleep(1)
	#cv2.waitKey(1000)
	ledoff()
# Store the calibration data
def store_data(data):

	f = open("calibration.txt", "wb")
	pickle.dump(data, f)
	f.close()


#Calculate and return homography matrix
def Calibrate_cam():

	print("Calibrating Camera...")

	ret, frame = cam.read()
	ref = cv2.imread('calb.jpg')

	ret1, corners1 = cv2.findChessboardCorners(ref, (7,4), None)
	ret2, corners2 = cv2.findChessboardCorners(frame, (7,4), None)

	if ret2 == True:
		h_mat, status = cv2.findHomography(corners2, corners1)
		ref = cv2.drawChessboardCorners(ref, (7,4), corners1,ret1)
		frame = cv2.drawChessboardCorners(frame, (7,4), corners2,ret2)
		cv2.imshow('result', frame)
		cv2.waitKey(0)
		cv2.destroyWindow('result')
		print(" Camera Calibrated ", h_mat)
		store_data(h_mat)
		return h_mat
	else:
		raise Exception("Chessboard not found")


def nothing(x):
	pass

#Main
if __name__=="__main__":
	pwm_angle = int(sys.argv[1])
	pwm_trans = int(sys.argv[2])
	cam = cv2.VideoCapture(0)

	#load calibration data
	f = open("calibration.txt", "rb")
	try:
		h = pickle.load(f)
	except:
		input("Calibration Required, Press enter when ready")
		h = Calibrate_cam()
	f.close()
	#cv2.namedWindow("Edge Detector")
	cv2.namedWindow("Shape Selection")
	# cv2.createTrackbar("Grad_X", "Edge Detector", 0, 255, nothing)
	# cv2.createTrackbar("Grad_Y", "Edge Detector", 0, 255, nothing)

	# cv2.createTrackbar("Circle Area", "Shape Selection", 0, 1000, nothing)
	# cv2.createTrackbar("Triangle Area", "Shape Selection", 0, 1000, nothing)
	# cv2.createTrackbar("Quadrilateral Area", "Shape Selection", 0, 1000, nothing)
	# cv2.createTrackbar("Min dist", "Shape Selection", 1, 1000, nothing)

	# Main Loop
	while(True):

		ret, frame = cam.read()
		# Apply Prespective Transform
		# cal_frame = frame
		cal_frame = cv2.warpPerspective(frame, h, (frame.shape[1], frame.shape[0]))
		#cal_frame *= 2
		k = cv2.waitKey(1)

		# Re-Calibration
		if k == ord('c'):
			try:
				h_old = h
				h = Calibrate_cam()
			except:
				h = h_old
				print("Calibration Failed")
				continue

		#quit
		if k == ord('x'):
			break

		# Manual Drive
		if k == ord('w'):
			botenable(60)
			botforward()
		elif k == ord('s'):
			botenable(60)
			botbackward()
		elif k == ord('a'):
			botenable(60)
			botleft()
		elif k == ord('d'):
			botenable(60)
			botright()
		elif k == ord('l'):
			blink_led()
		elif k == ord('e'):
			GPIO.cleanup()

		my_origin = (int(cal_frame.shape[1]/2), cal_frame.shape[0] - 1)
		cal_frame_process = cv2.cvtColor(cal_frame,cv2.COLOR_BGR2HSV)
		kernel = np.zeros((5, 5), np.uint8)
		
		#cal_frame_process = cv2.cvtColor(cal_frame_process, cv2.COLOR_BGR2GRAY)
		#cal_frame_process = cv2.GaussianBlur(cal_frame_process, (5, 5), 0)
		#cal_frame_process = cv2.adaptiveThreshold(cal_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
		grad_x = 150 #cv2.getTrackbarPos("Grad_X", "Edge Detector")
		grad_y = 150 #cv2.getTrackbarPos("Grad_Y", "Edge Detector")
		# circles = cv2.HoughCircles(cal_frame_process, cv2.HOUGH_GRADIENT, 1.2, minDist = 60, minRadius = 50)
		cal_frame_process = cv2.Canny(cal_frame_process, 100, 50)
		cal_frame_process = cv2.dilate(cal_frame_process, kernel, iterations = 5)
		cv2.imshow("Edes", cal_frame_process)
		_,contours,_= cv2.findContours(cal_frame_process, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		
		# lower = (0, 100, 0)
		# upper = (150, 255, 100)
		# mask = cv2.inRange(cal_frame, lower, upper)
		# cv2.imshow("mask", mask)

		# params = cv2.SimpleBlobDetector_Params()

		# # Change thresholds
		# # params.minThreshold = 10
		# # params.maxThreshold = 200 	


		# # Filter by Area.
		# params.filterByArea = True
		# params.minArea = 20

		# # Filter by Circularity
		# # params.filterByCircularity = True
		# # params.minCircularity = 0.8
		# detector = cv2.SimpleBlobDetector_create()


		# # Detect blobs.
		# if mask is not None:
		# 	keypoints = detector.detect(mask)
		# print(keypoints)
		# # Draw detected blobs as red circles.
		# # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
		# # the size of the circle corresponds to the size of blob

		# im_with_keypoints = cv2.drawKeypoints(cal_frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

		# # Show blobs
		# cv2.imshow("Keypoints", im_with_keypoints)


		closest_dis = pow(10, 9)
		closest_circle = None
		closest_traingle_dis = pow(10, 9)
		closest_quadrilateral_dis = pow(10, 9)
		triangle_centre = (0, 0)
		quadrilateral_centre = (0, 0)
		flag1 = 0
		flag2 = 0
		flag3 = 0
		for cnt in contours:
			approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
			# cv2.drawContours(cal_frame_gray, [approx], 0, (0), 5)
			x = approx.ravel()[0]
			y = approx.ravel()[1]

			if len(approx) == 3 and cv2.contourArea(cnt) > 250:
				rotatedRect = cv2.minAreaRect(cnt)
				area = cv2.contourArea(cnt)
				rect = cv2.boxPoints(rotatedRect)
				rect = np.int0(rect)
				cv2.drawContours(cal_frame,[rect], 0, ( 0, 0, 255), 2)
				rect_centroid = ((rect[0][0] + rect[1][0] + rect[2][0] + rect[3][0])/4, (rect[0][1] + rect[1][1] + rect[2][1] + rect[3][1])/4)
				cX = rect_centroid[0]
				cY = rect_centroid[1]
				if math.sqrt(math.pow(cX - my_origin[0], 2) + math.pow(cY - my_origin[1], 2)) < closest_traingle_dis:
					triangle_centre = (int(cX), int(cY))
					closest_traingle_dis = math.sqrt(math.pow(cX - my_origin[0], 2) + math.pow(cY - my_origin[1], 2))
					flag1 = 1
				# cv2.drawContours(cal_frame, [approx], 0, (0), 5)
				# M = cv2.moments(cal_frame_thresh)
				# cX = int(M["m10"] / M["m00"])
				# cY = int(M["m01"] / M["m00"])
				# closest_traingle_dis = min(closest_traingle_dis, math.sqrt(math.pow(cX - my_origin[0], 2) + math.pow(cY - my_origin[1], 2)))

			elif len(approx) == 4:    #or 5 maybe
				if len(approx) >= 4 and len(approx) <=5 and cv2.contourArea(cnt) > 250:    #or 5 maybe
					rotatedRect = cv2.minAreaRect(cnt)
					area = cv2.contourArea(cnt)
					rect = cv2.boxPoints(rotatedRect)
					rect = np.int0(rect)
					cv2.drawContours(cal_frame,[rect], 0, (255, 0, 0), 2)
					rect_centroid = ((rect[0][0] + rect[1][0] + rect[2][0] + rect[3][0])/4, (rect[0][1] + rect[1][1] + rect[2][1] + rect[3][1])/4)
					cX = rect_centroid[0]
					cY = rect_centroid[1]
					if math.sqrt(math.pow(cX - my_origin[0], 2) + math.pow(cY - my_origin[1], 2)) < closest_quadrilateral_dis:
						quadrilateral_centre = (int(cX),int(cY))
						flag2 = 1
						closest_quadrilateral_dis = math.sqrt(math.pow(cX - my_origin[0], 2) + math.pow(cY - my_origin[1], 2))
					# closest_quadrilateral_dis = min(closest_quadrilateral_dis, math.sqrt(math.pow(cX - my_origin[0], 2) + math.pow(cY - my_origin[1], 2)))
			elif len(approx) >= 8 and len(approx) < 28 and cv2.contourArea(cnt) > 100:
				if cv2.isContourConvex(approx):
					(cx, cy), radius = cv2.minEnclosingCircle(cnt)
					cv2.circle(cal_frame, (int(cx), int(cy)), int(radius), (0, 255, 0), 4)
					# M = cv2.moments(cnt)
					# cX = int(M["m10"] / M["m00"])
					# cY = int(M["m01"] / M["m00"])
					dis = math.sqrt(math.pow(my_origin[0] - cx, 2) + math.pow(my_origin[1] - cy, 2))
					if dis < closest_dis:
						closest_dis = dis
						flag3 = 1
						closest_circle = (cx, cy, radius)
		if not (flag1 or flag2 or flag3):
			botstop()
			print("STOPPPPPPPPP")
		
		if not closest_dis > 1000000:	
			# circles = np.round(circles[0, :]).astype("int")
			# for (x, y, r) in circles:
			# 	if r < 50:
			# 		continue
			# 	cv2.circle(cal_frame, (x, y), r, (0, 255, 0), 4)
			# 	dis = math.sqrt(math.pow(my_origin[0] - x, 2) + math.pow(my_origin[1] - y, 2))
			# 	if dis < closest_dis:
			# 		closest_dis = dis
			# 		closest_circle = (x, y, r)
			if closest_dis < closest_traingle_dis and closest_dis < closest_quadrilateral_dis:
				if closest_circle != None:
					direction = np.arctan2(closest_circle[1] - my_origin[1], -closest_circle[0] + my_origin[0])
					cv2.line(cal_frame, my_origin, (int(closest_circle[0]),int(closest_circle[1])), (0, 255, 255), 3)
					print(direction)
					if np.absolute(direction) > 1.65:
						# print("Move Left")
						botenable(pwm_angle)
						botright()
					elif np.absolute(direction) < 1.35:
						# print("Move Right")
						botenable(pwm_angle)
						botleft()
					else:
						# print("Move Straight")
						botenable(pwm_trans)
						botforward()
			elif closest_traingle_dis < closest_quadrilateral_dis:
				cv2.line(cal_frame, my_origin, triangle_centre, (0, 255, 255), 3)
				# print("Move Left")
				botenable(pwm_trans)
				botforward()
				time.sleep(1)
				#cv2.waitKey(1000)
				botenable(pwm_angle)
				botleft()
				time.sleep(1.5)
				#cv2.waitKey(1500)
				blink_led()
			elif closest_traingle_dis > closest_quadrilateral_dis:
				cv2.line(cal_frame, my_origin, quadrilateral_centre, (0, 255, 255),3)
				# print("Move Right")
				botenable(pwm_trans)
				botforward()
				time.sleep(1)
				#cv2.waitKey(1000)
				botenable(pwm_angle)
				botright()
				time.sleep(1.5)
				#cv2.waitKey(1500)
				blink_led()
				blink_led()
			else:
				botstop()
				# print("STOP")
		else:
			if closest_traingle_dis < closest_quadrilateral_dis:
				cv2.line(cal_frame, my_origin, triangle_centre, (0, 255, 255), 3)
				# print("Move Left")
				botenable(pwm_trans)
				botforward()
				time.sleep(1)
				#cv2.waitKey(1000)
				botenable(pwm_angle)
				botleft()
				time.sleep(1.5)
				#cv2.waitKey(1500)
				blink_led()
			elif closest_traingle_dis > closest_quadrilateral_dis:
				cv2.line(cal_frame, my_origin, quadrilateral_centre, (0, 255, 255),3)
				# print("Move Right")
				botenable(pwm_trans)
				botforward()
				time.sleep(1)
				#cv2.waitKey(1000)
				botenable(pwm_angle)
				botright()
				time.sleep(1.5)
				#cv2.waitKey(1500)
				blink_led()
				blink_led()
		#botstop()
		

		# Display Window
		cv2.imshow('Shape Selection',cal_frame)



	cam.release()
	cv2.destroyAllWindows()


