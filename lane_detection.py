import cv2
import numpy as np

def detect_edges(frame):
	
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	lower_lane = np.array([0, 0, 150])
	upper_lane= np.array([30, 30, 200])
	mask = cv2.inRange(hsv, lower_lane, upper_lane)

	edges = cv2.Canny(mask, 200, 400)

	return edges


def crop(edges):

    height, width = edges.shape
    mask = np.zeros_like(edges)

    # only focus bottom half of the screen
    polygon = np.array([[
        (0, height * 1 / 2),
        (width, height * 1 / 2),
        (width, height),
        (0, height),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    return cropped_edges

def detect_lines(cropped_edges):
    rho = 1
    angle = np.pi / 180 
    min_threshold = 10 
    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, np.array([]), minLineLength=8, maxLineGap=4)

    return line_segments

def average_slope_intercept(frame, line_segments):

    lane_lines = []
    if line_segments is None:
        print('No line_segment segments detected')
        return lane_lines

    height, width, _ = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1/3
    left_region_boundary = width * (1 - boundary)  # left lane line segment should be on left 2/3 of the screen
    right_region_boundary = width * boundary # right lane line segment should be on left 2/3 of the screen

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                print('skipping vertical line segment (slope=inf): %s' % line_segment)
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))

    print('lane lines: %s' % lane_lines)

    return lane_lines

def make_points(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height 
    y2 = int(y1 * 1 / 2) 

    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]

def display_lines(frame, lines, line_color=(0, 255, 0), line_width=2):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image


def find_path(frame):

	edges = detect_edges(frame)
	cropped = crop(edges)
	lines = detect_lines(cropped)
	lanes = average_slope_intercept(frame, lines)

	return lanes
	
# Main

cam = cv2.VideoCapture(0)

while (True) :

	ret, frame = cam.read()

	height, width, _ = frame.shape
	mid = int(width / 2)

	lanes = find_path(frame)
	draw_lanes = display_lines(frame, lanes)

	try:
		_, _, left_x2, _ = lanes[0][0]
		_, _, right_x2, _ = lanes[1][0]
		x_offset = int((left_x2 + right_x2)/2)
		y_offset = int(height/2)

	except:
		x1, _, x2, _ = lanes[0][0]
		x_offset = 2*x2 - x1
		y_offset = int(height / 2)

	cv2.line(draw_lanes, (mid, height), (x_offset, y_offset), (0, 0, 255), thickness=5, lineType=8)

	k = cv2.waitKey(1)
	if k == ord('x'):
		break
	# Display Window
	cv2.imshow('frame', draw_lanes)


# frame = cv2.imread('test.jpg')

# height, width, _ = frame.shape
# mid = int(width / 2)

# lanes = find_path(frame)
# draw_lanes = display_lines(frame, lanes)

# try:
# 	_, _, left_x2, _ = lanes[0][0]
# 	_, _, right_x2, _ = lanes[1][0]
# 	x_offset = (left_x2 + right_x2)/2
# 	y_offset = int(height/2)

# except:
# 	x1, _, x2, _ = lanes[0][0]
# 	x_offset = 2*x2 - x1
# 	y_offset = int(height / 2)

# print("X: ", x_offset, " Y: ", y_offset)
# cv2.line(draw_lanes, (mid, height), (int(x_offset), int(y_offset)), (0, 0, 255), thickness=5, lineType=8)

# # Display Window
# cv2.imshow('frame', draw_lanes)
# cv2.waitKey(0)
# cv2.waitKey(0)
# cv2.waitKey(0)