import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('test.jpg')
rows, cols = img.shape[:2]

M_translate = np.float32([[1, 0, 100],
                          [0, 1, 50]])
translated = cv2.warpAffine(img, M_translate, (cols, rows))

M_rotate = cv2.getRotationMatrix2D((cols//2, rows//2), 45, 1)
rotated = cv2.warpAffine(img, M_rotate, (cols, rows))

scaled = cv2.resize(img, None, fx=0.5, fy=0.5)

pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
M_affine = cv2.getAffineTransform(pts1, pts2)
affine = cv2.warpAffine(img, M_affine, (cols, rows))

pts1 = np.float32([[0, 0], [cols-1, 0], [0, rows-1], [cols-1, rows-1]])
pts2 = np.float32([[0, 0], [cols-1, 0], [100, rows-1], [cols-100, rows-1]])
M_perspective = cv2.getPerspectiveTransform(pts1, pts2)
perspective = cv2.warpPerspective(img, M_perspective, (cols, rows))

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
translated = cv2.cvtColor(translated, cv2.COLOR_BGR2RGB)
rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
scaled = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
affine = cv2.cvtColor(affine, cv2.COLOR_BGR2RGB)
perspective = cv2.cvtColor(perspective, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(15,10))

plt.subplot(2,3,1)
plt.imshow(img)
plt.title("Original")
plt.axis('off')

plt.subplot(2,3,2)
plt.imshow(translated)
plt.title("Translated")
plt.axis('off')

plt.subplot(2,3,3)
plt.imshow(rotated)
plt.title("Rotated")
plt.axis('off')

plt.subplot(2,3,4)
plt.imshow(scaled)
plt.title("Scaled")
plt.axis('off')

plt.subplot(2,3,5)
plt.imshow(affine)
plt.title("Affine")
plt.axis('off')

plt.subplot(2,3,6)
plt.imshow(perspective)
plt.title("Perspective")
plt.axis('off')

plt.tight_layout()
plt.show()