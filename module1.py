import cv2
import numpy as np

# Reading Video
cap = cv2.VideoCapture('5.mp4')

#Background Subtraction
mask = cv2.createBackgroundSubtractorMOG2(history=1, varThreshold=15, detectShadows=False)
#Making matrix for Erosion, dilation and morphing
kernel = np.ones((2,2),np.uint8)
kernel1 = np.ones((1,2),np.uint8)
#mask = cv2.dilate(mask, kernel, iterations=1)

#This is where the video is read
while cap.isOpened():


    ret, frame = cap.read()
    if not ret:
        break
    mask1 = mask.apply(frame)
    # Erosion
    mask1 = cv2.erode(mask1, kernel, iterations=1)
    # Dialtion
    mask1 = cv2.dilate(mask1, kernel1, iterations=3)
    # Morphing
    #mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)
    #mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if cv2.contourArea(c) < 750:
            continue
        elif cv2.contourArea(c) > 2000:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + int(w), y + int(h)), (255, 255, 255), 2)
        

    cv2.imshow('result', mask1)
    cv2.imshow('result', frame)
    k = cv2.waitKey(100) & 0xFF
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()

