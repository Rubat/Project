import cv2
import numpy as np

cap = cv2.VideoCapture('1.mp4')

mask = cv2.createBackgroundSubtractorMOG2()

arr = np.array([])
hull = []
while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        h = frame.shape[0]
        w = frame.shape[1]
        #print(w)

        mask1 = mask.apply(frame)

        _, contours, _ = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # calculate points for each contour
        for i in range(len(contours)):
            # creating convex hull object for each contour
            hull.append(cv2.convexHull(contours[i], False))


        for c in contours:
            if cv2.contourArea(c) < 500:
                continue
            elif cv2.contourArea(c) > 2500:
                (x, y, w, h) = cv2.boundingRect(c)
                rect = cv2.contourArea(c)
                drawing = np.zeros((mask1.shape[0], mask1.shape[1], 3), np.uint8)
                color_contours = (0, 255, 0) # green - color for contours
                color = (255, 0, 0) # blue - color for convex hull
                # draw ith contour
                cv2.drawContours(drawing, contours, i, color_contours, 1)
                # draw ith convex hull object
                cv2.drawContours(drawing, hull, i, color, 1, 8)

        cv2.imshow('Result', frame)
        k = cv2.waitKey(50) & 0xFF
        if k == 27:
            break


cap.release()
cv2.destroyAllWindows()