import cv2
import matplotlib.pyplot as plt

image = cv2.imread("test.jpg")

if image is None:
    print("Error: Image not found")
    exit()

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

resized_image = cv2.resize(image_rgb, (300, 300))

cropped_image = image_rgb[50:250, 100:300]

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(gray_image, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(resized_image)
plt.title("Resized Image")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(cropped_image)
plt.title("Cropped Image")
plt.axis("off")

plt.tight_layout()
plt.show()