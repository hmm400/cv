import cv2
import matplotlib.pyplot as plt

image_bgr = cv2.imread('test.jpg')

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

image_ycbcr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(image_rgb)
plt.title('RGB Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(image_hsv)
plt.title('HSV Image')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(image_lab)
plt.title('LAB Image')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(image_ycbcr)
plt.title('YCbCr Image')
plt.axis('off')

plt.tight_layout()
plt.show()