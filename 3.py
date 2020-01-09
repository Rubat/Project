import numpy as np
import cv2

#flags = [i for i in dir(cv2) if i.startswith('COLOR_G')]
#print(flags)

img = cv2.imread('1.jpg', 1)
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
#cv2.imshow('1', hsv)
#cv2.waitKey(50000) & 0xFF

# define range of red color in HSV
lower_red = np.array([110, 50, 50])
upper_red = np.array([130, 255, 255])

# define range of orange color in HSV
lower_orange = np.array([91, 50, 50])
upper_orange = np.array([111, 255, 255])

# define range of brown color in HSV
lower_brown = np.array([102, 50, 50])
upper_brown = np.array([122, 255, 255])

# define range of green color in HSV
lower_green = np.array([50, 100, 100])
upper_green = np.array([70, 255, 255])

# define range of blue color in HSV
lower_blue = np.array([154, 2, 21])
upper_blue = np.array([174, 255, 255])

# Threshold the HSV image to get only blue colors
mask_red = cv2.inRange(hsv, lower_red, upper_red)
mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
mask_green = cv2.inRange(hsv, lower_green, upper_green)
#mask = mask_red + mask_orange + mask_brown  + mask_blue

# Bitwise-AND mask and original image
res1 = cv2.bitwise_and(img,img, mask= mask_red)
res2 = cv2.bitwise_and(img,img, mask= mask_orange)
res3 = cv2.bitwise_and(img,img, mask= mask_brown)
res4 = cv2.bitwise_and(img,img, mask= mask_blue)
res5 = cv2.bitwise_and(img,img, mask= mask_green)
#res = res1 + res2 + res3 + res4

cv2.imshow('frame',img)
#cv2.imshow('frame',img)
#cv2.imshow('frame',img)
#cv2.imshow('frame',img)
#cv2.imshow('frame',img)

cv2.imshow('mask',mask_red)
#cv2.imshow('mask',mask_orange)
#cv2.imshow('mask',mask_brown)
#cv2.imshow('mask',mask_blue)
#cv2.imshow('mask',mask_green)

cv2.imshow('res',res1)
#cv2.imshow('res',res2)
#cv2.imshow('res',res3)
#cv2.imshow('res',res4)
#cv2.imshow('res',res5)


k = cv2.waitKey(0) & 0xFF
