import cv2
import numpy as np
import matplotlib.pyplot as plt 
from os import listdir
from os.path import isfile, join

img = cv2.imread('../data/input/qh.jpg', cv2.IMREAD_GRAYSCALE)

# First, we'll threshold the image and make it a binary image 
ret,imthresh = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)

# Perform opening on thresholded image to remove noise in space
kernel = np.ones((8,8), np.uint8)
imthresh_open = cv2.morphologyEx(imthresh, cv2.MORPH_OPEN, kernel)

# Get 4 corner points using rotated rectangle
_, contours, _ = cv2.findContours(imthresh_open,2,1)
#cv2.imshow("uh", imthresh_open)
cnt = contours[0]
#Find largest contour
cnt_max = 0
for contour in contours:
	if cv2.contourArea(contour) > cnt_max:
		cnt_max = cv2.contourArea(contour)
		cnt = contour
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)

cornerNW = box[1]
cornerNE = box[2]
cornerSW = box[0]
cornerSE = box[3]

## TODO
# To get 4 corners of a distorted image, we can use convex hull
hull = cv2.convexHull(cnt)
# Get furthest points of the hull in the 8 cardinal directions
# (Not sure the following is correct, should be verified)
# N: 	Maximize -y
# NE: 	Maximize x - y
# E: 	Maximize x
# SE: 	Maximize x + y
# S: 	Maximize y
# SW:	Maximize -x + y
# W: 	Maximize -x
# NW:	Maximize -x - y
# The resulting hull will be an octagon
# Take the 4 longest sides of and intersect them to find the 4 corners

# Another solution might involve approxPoly but I don't know how to use it
##

# Do some homography stuff to "straighten" the card
pts_src = np.array([cornerNW, cornerNE, cornerSW, cornerSE])
pts_dst = np.array([[0.0, 0.0], [691, 0.0], [0.0, 1056],[691, 1056]])
im_dst = np.zeros((1056, 691, 1), np.uint8)
homo, status = cv2.findHomography(pts_src, pts_dst)
im_homo = cv2.warpPerspective(imthresh, homo, (im_dst.shape[1],im_dst.shape[0]))
#cv2.imshow('homo', im_homo);

# Isolate the card letter and symbol
im_sym_letter = im_homo[18:600, 18:150]
ret2,im_sym_letter_thresh = cv2.threshold(im_sym_letter, 200, 255, cv2.THRESH_BINARY)
im_sym_letter_thresh_open = cv2.morphologyEx(im_sym_letter_thresh, cv2.MORPH_OPEN, kernel)
#cv2.imshow('im_sym_letter_thresh', im_sym_letter_thresh_open)

# Find letter and symbol bounding boxes
_, contours, _ = cv2.findContours(cv2.bitwise_not(im_sym_letter_thresh_open),2,1)

#Find largest contour for letter
cnt_letter = None
cnt_letter_max = 0
for contour in contours:
	if cv2.contourArea(contour) > cnt_letter_max:
		cnt_letter_max = cv2.contourArea(contour)
		cnt_letter = contour
contours.remove(cnt_letter)

#Find second largest contour for symbol
cnt_sym = None
cnt_sym_max = 0
for contour in contours:
	if cv2.contourArea(contour) > cnt_sym_max:
		cnt_sym_max = cv2.contourArea(contour)
		cnt_sym = contour

rect_sym = cv2.boundingRect(cnt_sym)
rect_letter = cv2.boundingRect(cnt_letter)

# Crop letter and symbol into two images, resize both to a set standard
im_sym = im_sym_letter_thresh[rect_sym[1]:rect_sym[1]+rect_sym[3], rect_sym[0]:rect_sym[0]+rect_sym[2]]
im_letter = im_sym_letter_thresh[rect_letter[1]:rect_letter[1]+rect_letter[3], rect_letter[0]:rect_letter[0]+rect_letter[2]]
im_sym_resized = cv2.resize(im_sym, (100,100))
im_letter_resized = cv2.resize(im_letter, (100,100))
ret3, im_sym_thresh = cv2.threshold(im_sym_resized, 200, 255, cv2.THRESH_BINARY)
ret4, im_letter_thresh = cv2.threshold(im_letter_resized, 200, 255, cv2.THRESH_BINARY)
# cv2.imshow('im_letter_thresh', im_letter_thresh)
# cv2.imshow('im_sym_thresh', im_sym_thresh)

# Search through all the template cards and find the one with the lowest difference
bestError = float("inf")
bestMatch = ""
mypath='../data/cards_png'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
for n in range(0, len(onlyfiles)):
	cur_img = cv2.imread( join(mypath,onlyfiles[n]), cv2.IMREAD_GRAYSCALE )
	# Isolate letter and symbol then threshold
	cur_sym_letter = cur_img[18:300, 18:110]
	ret4, cur_sym_letter_thresh = cv2.threshold(cur_sym_letter, 200, 255, cv2.THRESH_BINARY)
	# Find letter and symbol bounding boxes
	_, cur_contours, _ = cv2.findContours(cv2.bitwise_not(cur_sym_letter_thresh),2,1)
	cur_cnt_sym = cur_contours[0]
	cur_cnt_letter = cur_contours[1] 
	cur_rect_sym = cv2.boundingRect(cur_cnt_sym)
	cur_rect_letter = cv2.boundingRect(cur_cnt_letter)

	# Crop letter and symbol into two images, resize both to a set standard
	cur_sym = cur_sym_letter_thresh[cur_rect_sym[1]:cur_rect_sym[1]+cur_rect_sym[3], cur_rect_sym[0]:cur_rect_sym[0]+cur_rect_sym[2]]
	cur_letter = cur_sym_letter_thresh[cur_rect_letter[1]:cur_rect_letter[1]+cur_rect_letter[3], cur_rect_letter[0]:cur_rect_letter[0]+cur_rect_letter[2]]
	cur_sym_resized = cv2.resize(cur_sym, (100,100))
	cur_letter_resized = cv2.resize(cur_letter, (100,100))
	ret5, cur_sym_thresh = cv2.threshold(cur_sym_resized, 200, 255, cv2.THRESH_BINARY)
	ret6, cur_letter_thresh = cv2.threshold(cur_letter_resized, 200, 255, cv2.THRESH_BINARY)

	# Find difference between input card and template card
	out_sym = im_sym_thresh - cur_sym_thresh
	out_letter = im_letter_thresh - cur_letter_thresh
	# if onlyfiles[n] == "QH.png":
	# 	cv2.imshow("cur_sym_thresh", cur_sym_thresh)
	# 	cv2.imshow("cur_letter_thresh", cur_letter_thresh)

	error = cv2.countNonZero(out_sym) + cv2.countNonZero(out_letter)
	if error < bestError:
		bestError = error
		bestMatch = onlyfiles[n]
print(bestMatch)

# Reminder on how to rotate images, in case I need it
#rotation_matrix = cv2.getRotationMatrix2D((w/2, h/2), 90, 1)
#img_rotation = cv2.warpAffine(imgcrop, rotation_matrix, (w, h))

cv2.waitKey(0)
cv2.destroyAllWindows()
