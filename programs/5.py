import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('test.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

kernel = np.ones((5,5), np.uint8)

erosion = cv2.erode(thresh, kernel, iterations=1)
dilation = cv2.dilate(thresh, kernel, iterations=1)

ret2, thresh2 = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

opening = cv2.morphologyEx(
    thresh2,
    cv2.MORPH_OPEN,
    kernel
)

markers = cv2.connectedComponents(opening)[1]
markers = cv2.watershed(img, markers)

watershed = img.copy()
watershed[markers == -1] = [255, 0, 0]

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
watershed_rgb = cv2.cvtColor(watershed, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(15,10))

plt.subplot(2,3,1)
plt.imshow(img_rgb)
plt.title("Original")
plt.axis('off')

plt.subplot(2,3,2)
plt.imshow(thresh, cmap='gray')
plt.title("Thresholding")
plt.axis('off')

plt.subplot(2,3,3)
plt.imshow(erosion, cmap='gray')
plt.title("Erosion")
plt.axis('off')

plt.subplot(2,3,4)
plt.imshow(dilation, cmap='gray')
plt.title("Dilation")
plt.axis('off')

plt.subplot(2,3,5)
plt.imshow(opening, cmap='gray')
plt.title("Opening")
plt.axis('off')

plt.subplot(2,3,6)
plt.imshow(watershed_rgb)
plt.title("Watershed")
plt.axis('off')

plt.tight_layout()
plt.show()