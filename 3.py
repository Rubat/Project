#Importing the Libraries
import cv2
import numpy as np
from skimage import measure
from math import floor

#Reading 2 Frames
image1 = cv2.imread('1.jpg')
image2 = cv2.imread('190.jpg')

#Converting the Frames into Gray Scale
image3 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image4 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

#Subtracting The Frames
sub = cv2.subtract(image3, image4)

#Binary threshing of Subtracted Image
ret1, th1 = cv2.threshold(sub, 65, 255, cv2.THRESH_BINARY)

#Find Connected pixels in the Threshed Image
thresh = th1
labels = measure.label(thresh, neighbors=8, background=0)
mask = np.zeros(thresh.shape, dtype="uint8")

#Finding Pixels having Non-Zero Value
for label in np.unique(labels):
    if label == 0:
        continue
    labelmask = np.zeros(thresh.shape, dtype="uint8")
    labelmask[labels == label] = 255
    numPixels = cv2.countNonZero(labelmask)

    #Setting the Value of Connected Pixels in the Threshed Image
    if numPixels > 700:
        mask = cv2.add(mask, labelmask)

#Resolution of Image
height = mask.shape[0]
width = mask.shape[1]

#Creating two new arrays
arr = np.array([])
arr1 = np.array([])

#Finding the Index of Rows and Columns having value of 255
for c in range(0, height):
    for d in range(0, width):
        if mask[c, d] == 255:
            arr = np.append(arr, [c])
            arr1 = np.append(arr1, [d])

#For Rows
arr = np.unique(arr)
#print(arr)
a = arr.size
#Bottom Left
w = (np.amax(arr))
w = floor(w)
#Left Top
y = (np.amin(arr1))
y = floor(y)

#For Columns
arr1 = np.unique(arr1)
#print(arr1)
b = arr1.size
#Right Top
h = (np.amax(arr1))
h = floor(h)
#Top
x = (np.amin(arr))
x = floor(x)

#expansion of rectangle
e = 25
f = e+1

hh = (h-y)
ww = (w-x)

#Creating the Rectangle along the Detected Area
cv2.rectangle(image1, (y-f, x-f), (h+e, w+e), (0, 0, 0), 1)

#Rectangle width
q = y-f
t = h+e
Rectangle_width = t-q
print(Rectangle_width)

#Rectangle Height
r = x-f
u = w+e
if u > 250:
    u = 230
Rectangle_Height = u-r
print(Rectangle_Height)

#Crop the rectangle
#cropped = image1[x-e:w+e, y-e:h+e]

roi = image1[r:u, q:t]
black_bg = 0*np.ones_like(image1)

#cv2.rectangle(res, (x, y), (x + w, y + h), (0, 0, 0), 1)
#--- paste ROIs on image with white background
black_bg[r:u, q:t] = roi

image1 = black_bg

#Editing Image1
new_image = np.zeros(image1.shape, image1.dtype)

alpha = 1.5
beta = 42
gamma = 1.7

#new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

lookUpTable = np.empty((1, 256), np.uint8)

for i in range(256):
    lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

res = cv2.LUT(image1, lookUpTable)

hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)

lower_red = np.array([175, 150, 48])
upper_red = np.array([185, 255, 255])
mask = cv2.inRange(hsv, lower_red, upper_red)

#Finding Contours
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

#Creating Rectangle around the the Contours
for c in contours:
    # get the bounding rect
    x, y, w, h = cv2.boundingRect(c)
    # get the min area rect
    rect = cv2.minAreaRect(c)
    #Condition Width should be greater or equal to 10 Pixels
    if rect[1][1] >= 10:
        #Creating the Rectangle
        cv2.rectangle(res, (x, y), (x + w, y + h), (0, 0, 255), -1)

#Printing the Result
cv2.imshow("Mask", res)
cv2.waitKey(0)
