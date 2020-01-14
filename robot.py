import cv2
import numpy as np
import pickle


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

	#Processing


	# Display Window
	cv2.imshow('frame',cal_frame)



cam.release()
cv2.destroyAllWindows()


