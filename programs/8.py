import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('test.jpg', 0)
img2 = cv2.imread('test.jpg', 0)

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

result1 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None)

sift = cv2.SIFT_create()

kp3, des3 = sift.detectAndCompute(img1, None)
kp4, des4 = sift.detectAndCompute(img2, None)

index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches2 = flann.knnMatch(des3, des4, k=2)

good = []
for m, n in matches2:
    if m.distance < 0.75 * n.distance:
        good.append(m)

result2 = cv2.drawMatches(img1, kp3, img2, kp4, good[:20], None)

result1 = cv2.cvtColor(result1, cv2.COLOR_BGR2RGB)
result2 = cv2.cvtColor(result2, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.imshow(result1)
plt.title("BF Matching (ORB)")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(result2)
plt.title("FLANN Matching (SIFT)")
plt.axis('off')

plt.tight_layout()
plt.show()