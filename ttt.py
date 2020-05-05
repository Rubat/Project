import cv2
import numpy as np
import math

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
            # print(len(contours))
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

            frame = frame[y:y+h, x:x+w]
            roi = frame

            #find contours
            contours,hierarchy= cv2.findContours(mask1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

            #find contour of max area(hand)
            cnt = max(contours, key = lambda x: cv2.contourArea(x))

            #approx the contour a little
            epsilon = 0.0005*cv2.arcLength(cnt,True)
            approx= cv2.approxPolyDP(cnt,epsilon,True)

            #make convex hull around hand
            hull = cv2.convexHull(cnt)

            #define area of hull and area of hand
            areahull = cv2.contourArea(hull)
            areacnt = cv2.contourArea(cnt)

            #find the percentage of area not covered by hand in convex hull
            arearatio=((areahull-areacnt)/areacnt)*100

            #find the defects in convex hull with respect to hand
            hull = cv2.convexHull(approx, returnPoints=False)
            defects = cv2.convexityDefects(approx, hull)

            # l = no. of defects
            l=0

            #code for finding no. of defects due to fingers
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = tuple(approx[s][0])
                end = tuple(approx[e][0])
                far = tuple(approx[f][0])
                pt= (100,180)


                # find length of all sides of triangle
                a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                s = (a+b+c)/2
                ar = math.sqrt(s*(s-a)*(s-b)*(s-c))

                #distance between point and convex hull
                d=(2*ar)/a

                # apply cosine rule here
                angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57


                # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
                if angle <= 90 and d>30:
                    l += 1
                    cv2.circle(roi, far, 3, [255,0,0], -1)

                #draw lines around hand
                cv2.line(roi,start, end, [0,255,0], 2)


            l+=1

            #print corresponding gestures which are in their ranges
            font = cv2.FONT_HERSHEY_SIMPLEX
            if l==1:
                if areacnt<2000:
                    cv2.putText(frame,'Put hand in the box',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                else:
                    if arearatio<12:
                        cv2.putText(frame,'0',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    elif arearatio<17.5:
                        cv2.putText(frame,'Best of luck',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

                    else:
                        cv2.putText(frame,'1',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

            elif l==2:
                cv2.putText(frame,'2',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

            elif l==3:

                if arearatio<27:
                    cv2.putText(frame,'3',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                else:
                    cv2.putText(frame,'ok',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

            elif l==4:
                cv2.putText(frame,'4',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

            elif l==5:
                cv2.putText(frame,'5',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

            elif l==6:
                cv2.putText(frame,'reposition',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

            else :
                cv2.putText(frame,'reposition',(10,50), font, 2, (0,0,255), 3, cv2.LINE_AA)


            # mask3 = np.zeros(frame.shape, dtype=np.uint8)
            # cv2.fillPoly(mask3, pts=[c], color=(255,255,255))
            #
            # #apply the mask
            # masked_image = cv2.bitwise_and(frame, mask3)

            # upper_left = (x, y)
            # bottom_right = (x + w, y + h)
            # frame1 = frame[upper_left[1]: bottom_right[1], upper_left[0]: bottom_right[0]]
            # # roi = frame[y:y+h, x:x+w]
            # # black_bg = 0 * np.ones_like(frame)
            # # black_bg[y:y+h, x:x+w] = roi
            #
            # blurred_frame = cv2.GaussianBlur(frame1, (5, 5), 0)
            # # hsv1 = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2RGB)
            # hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_RGB2HSV)
            #
            # lower_red = np.array([110, 50, 50])
            # upper_red = np.array([130, 255, 255])
            # mask2 = cv2.inRange(hsv, lower_red, upper_red)
            #
            # # Erosion
            # mask2 = cv2.erode(mask2, kernel, iterations=4)
            # # Dilation
            # mask2 = cv2.dilate(mask2, kernel1, iterations=6)
            # # Morphing
            # # mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
            # # mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
            #
            # mask3 = np.zeros(frame.shape, dtype=np.uint8)
            # cv2.fillPoly(mask3, pts=[c], color=(255, 255, 255))
            # masked_image = cv2.bitwise_and(frame, mask3)

        # contours, _ = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        #
        # for contour in contours:
        #     cv2.drawContours(frame1, contour, -1, (0, 0, 0), 1)
        #
        # for b in contours:
        #     if cv2.contourArea(b) <250:
        #         continue
        #     # get the bounding rect
        #     elif cv2. contourArea(b) >350:
        #         x, y, w, h = cv2.boundingRect(b)
        #         # draw a green rectangle to visualize the bounding rect2
        #         cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 1)
        #         mask3 = np.zeros(frame.shape, dtype=np.uint8)
        #         cv2.fillPoly(mask3, pts=[c], color=(255, 255, 255))
        #
        #         # print(cv2.contourArea(b))
        #
        #         # apply the mask
        #         masked_image = cv2.bitwise_and(frame, mask3)

    #show the windows
    cv2.imshow('mask',mask)
    cv2.imshow('frame',frame)
    # cv2.imshow('result', mask1)
    # cv2.imshow('result', mask2)
    # cv2.imshow('result', mask3)
    # cv2.imshow('result', masked_image)
    # cv2.imshow('result', frame)
    # cv2.imshow('result', dst)
    k = cv2.waitKey(50) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
