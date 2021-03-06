import cv2
import numpy as np

# Reading Video
cap = cv2.VideoCapture('video.mp4')

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
        elif cv2.contourArea(c) > 1000:
            # print(len(contours))
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

            #Red Color Ranges
            low_red = np.array([161, 115, 84])
            high_red = np.array([179, 255, 255])
            red_mask = cv2.inRange(hsv, low_red, high_red)

            #Brown Color Ranges
            low_brown= np.array([5, 100, 50])
            high_brown= np.array([17, 255, 255])
            brown_mask = cv2.inRange(hsv, low_brown, high_brown)

            # Green color
            low_green = np.array([38, 100, 100])
            high_green = np.array([75, 255, 255])
            green_mask = cv2.inRange(hsv, low_green, high_green)

            # Orange color
            low_orange = np.array([5, 100, 100])
            high_orange = np.array([15, 255, 255])
            orange_mask = cv2.inRange(hsv, low_orange, high_orange)

            # Purple color
            low_purple = np.array([120, 100, 100])
            high_purple = np.array([165, 255, 255])
            purple_mask = cv2.inRange(hsv, low_purple, high_purple)

            #Every color except white
            low = np.array([0, 42, 0])
            high = np.array([179, 255, 255])
            umask = cv2.inRange(hsv, low, high)

            mask2 = purple_mask | orange_mask | green_mask | brown_mask | red_mask

            # Erosion
            mask2 = cv2.erode(mask2, kernel, iterations=4)
            # Dilation
            mask2 = cv2.dilate(mask2, kernel1, iterations=6)
            # Morphing
            # mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
            # mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)

            mask3 = np.zeros(frame.shape, dtype=np.uint8)
            cv2.fillPoly(mask3, pts=[c], color=(255, 255, 255))
            masked_image = cv2.bitwise_and(frame, mask3)

            contours, _ = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            # for contour in contours:
            #     cv2.drawContours(frame1, contour, -1, (0, 0, 0), 1)

            # m = cv2.moments(contour)
            # if m["m00"] != 0:
            #     cX= int(m["m10"] / m["m00"])
            #     cY= int(m["m01"] / m["m00"])
            # else:
            #     cX=0
            #     cY=0
            #
            # cv2.circle(frame, (cX, cY), 1, (0, 0, 0), -1)
            # cv2.putText(frame, "center", (cX - 20, cY - 20),
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


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

                    m = cv2.moments(b)
                    if m["m00"] != 0:
                        cX= int(m["m10"] / m["m00"])
                        cY= int(m["m01"] / m["m00"])
                    else:
                        cX=0
                        cY=0

                    cv2.circle(frame, (cX, cY), 2, (0, 0, 0), -1)

                    # print(cv2.contourArea(b))

                    # apply the mask
                    masked_image = cv2.bitwise_and(frame, mask3)


    # cv2.imshow('result', mask1)
    # cv2.imshow('result', mask2)
    # cv2.imshow('result', mask3)
    # cv2.imshow('result', masked_image)
    cv2.imshow('result', frame)
    # cv2.imshow('result', dst)
    k = cv2.waitKey(50) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
