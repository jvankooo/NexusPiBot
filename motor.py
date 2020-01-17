import RPi.GPIO as GPIO
from time import sleep
import cv2
 
GPIO.setmode(GPIO.BOARD)
 
Motor1a = 35    # Input Pin
Motor1b = 37    # Input Pin
Motor1e = 31	# Enable 1
Motor2a = 38    # Input Pin
Motor2b = 40    # Input Pin
Motor2e = 33	# Enable 2
 
GPIO.setup(Motor1a,GPIO.OUT)
GPIO.setup(Motor1b,GPIO.OUT)
GPIO.setup(Motor1e,GPIO.OUT)
GPIO.setup(Motor2a,GPIO.OUT)
GPIO.setup(Motor2b,GPIO.OUT)
GPIO.setup(Motor2e,GPIO.OUT)

cam = cv2.VideoCapture(0)

def botenable():
	GPIO.output(Motor1e, GPIO.HIGH)
	GPIO.output(Motor2e, GPIO.HIGH)

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
	GPIO.output(Motor1e, GPIO.LOW)
	GPIO.output(Motor2e, GPIO.LOW)

while (True):
	
	ret, frame = cam.read()
	k = cv2.waitKey(5)
	if k == ord('w'):
		botenable()
		botforward()
	elif k == ord('s'):
		botenable()
		botbackward()
	elif k == ord('a'):
		botenable()
		botleft()
	elif k == ord('d'):
		botenable()
		botright()
	elif k == ord('e'):
		GPIO.cleanup()
	else:
		botstop()
		
	cv2.imshow('vid', frame)
