import cv2
import numpy as np

cap = cv2.VideoCapture('1.mp4')

fgbg = cv2.createBackgroundSubtractorMOG2()

while 1:
    ret, frame = cap.read()
    mask = fgbg.apply(frame)

    ret1, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

    _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        rect = cv2.minAreaRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

        roi = frame[y:y+h, x:x+w]
        #black_bg = 0 * np.ones_like(frame)
        #black_bg[y:y+h, x:x+w] = roi

    cv2.imshow('Result', mask)
    k = cv2.waitKey(50) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()