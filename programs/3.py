import cv2
import matplotlib.pyplot as plt

image = cv2.imread('test.jpg')

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gaussian = cv2.GaussianBlur(image_rgb, (5, 5), 0)
median = cv2.medianBlur(image_rgb, 5)
bilateral = cv2.bilateralFilter(image_rgb, 9, 75, 75)

hist_eq = cv2.equalizeHist(gray)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe_img = clahe.apply(gray)

plt.figure(figsize=(15,10))

plt.subplot(2,3,1)
plt.imshow(image_rgb)
plt.title("Original")
plt.axis('off')

plt.subplot(2,3,2)
plt.imshow(gaussian)
plt.title("Gaussian")
plt.axis('off')

plt.subplot(2,3,3)
plt.imshow(median)
plt.title("Median")
plt.axis('off')

plt.subplot(2,3,4)
plt.imshow(bilateral)
plt.title("Bilateral")
plt.axis('off')

plt.subplot(2,3,5)
plt.imshow(hist_eq, cmap='gray')
plt.title("Histogram Equalization")
plt.axis('off')

plt.subplot(2,3,6)
plt.imshow(clahe_img, cmap='gray')
plt.title("CLAHE")
plt.axis('off')

plt.tight_layout()
plt.show()