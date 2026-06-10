import cv2
import matplotlib.pyplot as plt

img = cv2.imread('test.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray, None)
img_sift = cv2.drawKeypoints(
    img, kp1, None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

orb = cv2.ORB_create()
kp2, des2 = orb.detectAndCompute(gray, None)
img_orb = cv2.drawKeypoints(
    img, kp2, None,
    color=(0, 255, 0)
)

img_sift = cv2.cvtColor(img_sift, cv2.COLOR_BGR2RGB)
img_orb = cv2.cvtColor(img_orb, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.imshow(img_sift)
plt.title("SIFT Keypoints")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(img_orb)
plt.title("ORB Keypoints")
plt.axis('off')

plt.tight_layout()
plt.show()