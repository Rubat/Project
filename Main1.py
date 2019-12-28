import cv2
import numpy as np
import math

cap = cv2.VideoCapture('1.mp4')

mask = cv2.createBackgroundSubtractorMOG2()


arr0 = np.array([])
arr1 = np.array([])
# arr2 = np.array([])
# arr3 = np.array([])

ret, frame = cap.read()
h, w, c = frame.shape

x = 1
y = 1
w = 100
h = 125
f = 0.5

# while cap.isOpened():
# #
# #     ret, frame = cap.read()
# #     if not ret:
# #         break
# #
# #     mask1 = mask.apply(frame)
# #
# #     _, contours, _ = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #     #
# #     for c in contours:
# #         if cv2.contourArea(c) > 2500 and cv2.contourArea(c) < 10000:
# #             x, y, w, h = cv2.boundingRect(c)
# #     #         arr0 = np.append(arr0, x) #arr0[first - 1]
# #     #         arr1 = np.append(arr1, y)
# #             arr0 = np.append(arr0, h)
# #             arr1 = np.append(arr1, w)
# #
# # cap.release()
# #
# cap = cv2.VideoCapture('1.mp4')
#
# mask = cv2.createBackgroundSubtractorMOG2()


# print(arr0)
# print(arr1)
# print(sum(arr0)/len(arr0))
# ah = math.floor(sum(arr0)/len(arr0))
# print(sum(arr1)/len(arr1))
# aw = math.floor(sum(arr1)/len(arr1))
# print(len(arr0))
# print(len(arr0))

while cap.isOpened():

    # print(aw)
    ret, frame = cap.read()
    if not ret:
        break

    mask1 = mask.apply(frame)

    _, contours, _ = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if cv2.contourArea(c) < 500:
            continue
        elif cv2.contourArea(c) > 2500 and cv2.contourArea(c) <10000:

            # px = previous x .... py = previous y
            # px = x
            # py = y
            # pw = previous width .... ph = previous height
            pw = w
            ph = h
            # print(x,y)
            # print("Second")
            # pcw = previous centre width .... pch = previous centre height
            # pcw = math.ceil(w/2)
            # pch = math.ceil(h/2)
            (x, y, w, h) = cv2.boundingRect(c)
            # print(x,y)

            # wx  = weighted x ... wy = weighted y.... ww = weighted width ... wh = weighted height
            # x = (px + x) / 2
            # y = (py + y) / 2
            # wx = (1 - f) * px + f * x
            # wy = (1 - f) * py + f * y
            # w = (pw + w) / 2
            # h = (ph + h) / 2
            ww = (1 - f) * pw + f * w
            if ww > 105:
                ww = 105
            wh = (1 - f) * ph + f * h
            if wh > 135:
                wh = 135
            # ww = 105
            # wh = 135
            # print(ww)
            # print(wh)

            # nx =  new x .... ny = new y
            # nx = math.floor((px + x) / 2)
            # ny = math.floor((py + y) / 2)

            # ncw = new centre width ... nch = new centre height
            # ncw = math.ceil(w/2)
            # nch = math.ceil(h/2)
            #
            # #ach = average centre height ....... acw = average centre width
            # # acw = math.floor((pcw + ncw) / 2)
            # # ach = math.floor((pch + nch) / 2)
            # acw = pcw + ncw
            # ach = pch + nch

            # cv2.rectangle(frame, (x, y), (x + int(w), y + int(h)), (255, 255, 255), 2)
            # cv2.rectangle(frame, (int(x), int(y)), (int(x) + int(w), int(y) + int(h)), (255, 255, 255), 2)
            cv2.rectangle(frame, (x, y), (x + int(ww), y + int(wh)), (255, 255, 255), 2)
            # cv2.rectangle(frame, (int(wx), int(wy)), (int(wx) + int(ww), int(wy) + int(wh)), (255, 255, 255), 2)

    cv2.imshow('Result', frame)
    k = cv2.waitKey(50) & 0xFF
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()






