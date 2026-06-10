import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('test.jpg')

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.convertScaleAbs(sobely)

canny = cv2.Canny(gray, 100, 200)

lines = cv2.HoughLinesP(
    canny,
    1,
    np.pi / 180,
    threshold=100,
    minLineLength=50,
    maxLineGap=10
)

hough_image = image_rgb.copy()

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(hough_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

harris_gray = np.float32(gray)
corners = cv2.cornerHarris(harris_gray, 2, 3, 0.04)

harris_image = image_rgb.copy()
harris_image[corners > 0.01 * corners.max()] = [255, 0, 0]

plt.figure(figsize=(18,10))

plt.subplot(2,3,1)
plt.imshow(image_rgb)
plt.title("Original")
plt.axis('off')

plt.subplot(2,3,2)
plt.imshow(sobelx, cmap='gray')
plt.title("Sobel X")
plt.axis('off')

plt.subplot(2,3,3)
plt.imshow(sobely, cmap='gray')
plt.title("Sobel Y")
plt.axis('off')

plt.subplot(2,3,4)
plt.imshow(canny, cmap='gray')
plt.title("Canny Edge")
plt.axis('off')

plt.subplot(2,3,5)
plt.imshow(hough_image)
plt.title("Hough Lines")
plt.axis('off')

plt.subplot(2,3,6)
plt.imshow(harris_image)
plt.title("Harris Corners")
plt.axis('off')

plt.tight_layout()
plt.show()