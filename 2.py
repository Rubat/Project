import cv2
import numpy as np

means = 10
n = 1
    
image = cv2.imread("3.jpg", 1)
height, width, channels = image.shape
samples = np.zeros([height * width, 3], dtype=np.float32)
count = 0

for x in range(height):
    for y in range(width):
        samples[count] = image[x][y]  # BGR color
        count += 1

compactness, labels, centers = cv2.kmeans(samples, means, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001), n, cv2.KMEANS_RANDOM_CENTERS)
centers = np.uint8(centers)
res = centers[labels.flatten()]
image2 = res.reshape(image.shape)

cv2.imshow("KMEANS", image2)
cv2.imwrite("10-kmeans.jpg", image2)
cv2.waitKey(0)
cv2.destroyAllWindows()
