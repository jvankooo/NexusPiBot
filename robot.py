import cv2
import numpy as np
import pickle
import math


# Upon Transfer to Pi change all keys to GPIO buttons

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




#Main

cam = cv2.VideoCapture(0)

#load calibration data
f = open("calibration.txt", "rb")
try:
	h = pickle.load(f)
except:
	input("Calibration Required, Press enter when ready")
	h = Calibrate_cam()
f.close()


# Main Loop
while(True):

	ret, frame = cam.read()
	# Apply Prespective Transform

	cal_frame = cv2.warpPerspective(frame, h, (frame.shape[1], frame.shape[0]))

	# Re-Calibration
	if cv2.waitKey(1) & 0xFF == ord('c'):
		try:
			h_old = h
			h = Calibrate_cam()
		except:
			h = h_old
			print("Calibration Failed")
			continue

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	cal_frame_gray = cv2.cvtColor(cal_frame, cv2.COLOR_BGR2GRAY)
	cal_frame_blur = cv2.GaussianBlur(cal_frame_gray, (5, 5), 0)
	cal_frame_thresh = cv2.adaptiveThreshold(cal_frame_blur, 255, 1, 1, 11, 2)
	_, contours, _ = cv2.findContours(cal_frame_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	

	circles = cv2.HoughCircles(cal_frame, cv2.HOUGH_GRADIENT, 1.2, 100)

	closest_traingle_dis = pow(10, 9)
	closest_quadrilateral_dis = pow(10, 9)
	for cnt in contours:
		approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
		cv2.drawContours(cal_frame_gray, [approx], 0, (0), 5)
		x = approx.ravel()[0]
		y = approx.ravel()[1]

		if len(approx) == 3:
			M = cv2.moments(c)
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
			closest_traingle_dis = min(closest_traingle_dis, math.sqrt(math.pow(cX - my_origin[0], 2) + math.pow(cY - my_origin[1], 2)))

		elif len(approx) == 4:    #or 5 maybe
			M = cv2.moments(c)
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
			closest_quadrilateral_dis = min(closest_quadrilateral_dis, math.sqrt(math.pow(cX - my_origin[0], 2) + math.pow(cY - my_origin, 2)))


	my_origin = (cal_frame.shape[0] - 1, cal_frame.shape[1]/2)
	
	if circles is not None:
		closest_dis = pow(10, 9)
		closest_circle = None
		circles = np.round(circles[0, :]).astype("int")
		for (x, y, r) in circles:
			dis = math.sqrt(math.pow(my_origin[0] - x, 2) + math.pow(my_origin[1] - y, 2))
			if dis < closest_dis:
				closest_dis = dis
				closest_circle = (x, y, r)
		if closest_dis < closest_traingle_dis and closest_dis < closest_quadrilateral_dis:
			if closest_circle != None:
				direction = np.arctan2(my_origin[0] - closest_circle[0], closest_circle[1] - my_origin[1])
				if direction > 7 * np.pi/12:
					print("Move Left")
				elif direction < 5*np.pi/12:
					print("Move Right")
				else:
					print("Move Straight")
		elif closest_traingle_dis < closest_quadrilateral_dis:
			print("Move Left")
		else:
			print("Move Right")
	else:
		if closest_traingle_dis < closest_quadrilateral_dis:
			print("Move Left")
		else:
			print("Move Right")


	# Display Window
	cv2.imshow('frame',cal_frame)



cam.release()
cv2.destroyAllWindows()

