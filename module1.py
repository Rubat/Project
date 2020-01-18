import cv2
import numpy as np

# Reading Video
cap = cv2.VideoCapture('1.mp4')

# Background Subtraction
mask = cv2.createBackgroundSubtractorMOG2(history=1, varThreshold=15, detectShadows=False)
# Making matrix for Erosion, dilation and morphing
kernel = np.ones((2, 2), np.uint8)
kernel1 = np.ones((1, 2), np.uint8)

# This is where the video is read
while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break
    frame1 = frame

    mask1 = mask.apply(frame)
    # Erosion
    mask1 = cv2.erode(mask1, kernel, iterations=1)
    # Dialtion
    mask1 = cv2.dilate(mask1, kernel1, iterations=3)
    # Morphing
    # mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if cv2.contourArea(c) < 750:
            continue
        elif cv2.contourArea(c) > 2000:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

            dst = np.zeros_like(frame)
            dst[y:y+h, x:x+w] = frame[y:y+h, x:x+w]

            # mask3 = np.zeros(frame.shape, dtype=np.uint8)
            # cv2.fillPoly(mask3, pts=[c], color=(255,255,255))
            #
            # #apply the mask
            # masked_image = cv2.bitwise_and(frame, mask3)

            upper_left = (x, y)
            bottom_right = (x + w, y + h)
            frame1 = frame[upper_left[1]: bottom_right[1], upper_left[0]: bottom_right[0]]
            # roi = frame[y:y+h, x:x+w]
            # black_bg = 0 * np.ones_like(frame)
            # black_bg[y:y+h, x:x+w] = roi

            blurred_frame = cv2.GaussianBlur(frame1, (5, 5), 0)
            # hsv1 = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2RGB)
            hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_RGB2HSV)

            lower_red = np.array([110, 50, 50])
            upper_red = np.array([130, 255, 255])
            mask2 = cv2.inRange(hsv, lower_red, upper_red)

            # Erosion
            mask2 = cv2.erode(mask2, kernel, iterations=4)
            # Dilation
            mask2 = cv2.dilate(mask2, kernel1, iterations=6)
            # Morphing
            # mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
            # mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            for contour in contours:
                cv2.drawContours(frame1, contour, -1, (0, 0, 0), 1)

            for b in contours:
                if cv2.contourArea(b) <250:
                    continue
                # get the bounding rect
                elif cv2. contourArea(b) >350:
                    x, y, w, h = cv2.boundingRect(b)
                    # draw a green rectangle to visualize the bounding rect2
                    cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    mask3 = np.zeros(frame.shape, dtype=np.uint8)
                    cv2.fillPoly(mask3, pts=[c], color=(255, 255, 255))

                    # print(cv2.contourArea(b))

                    # apply the mask
                    masked_image = cv2.bitwise_and(frame, mask3)


    # cv2.imshow('result', mask1)
    # cv2.imshow('result', mask2)
    # cv2.imshow('result', mask3)
    # cv2.imshow('result', masked_image)
    # cv2.imshow('result', frame)
    cv2.imshow('result', dst)
    k = cv2.waitKey(100) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
