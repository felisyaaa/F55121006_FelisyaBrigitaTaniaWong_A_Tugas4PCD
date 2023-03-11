# F55121006_Felisya Brigita Tania Wong_A

import cv2
import numpy as np
import matplotlib.pyplot as plt

# membaca gambar
img = cv2.imread('bloodCells.jpg')

# konversi gambar ke grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# menerapkan median filter
median_img = cv2.medianBlur(gray_img, 3)

# menerapkan min filter
kernel = np.ones((3,3), np.uint8)
min_img = cv2.erode(gray_img, kernel)

# menerapkan max filter
max_img = cv2.dilate(gray_img, kernel)

# menampilkan gambar
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 10))
axs[0, 0].text(0.5, 0.5, 'Median Filter, Min Filter & Max Filter', ha='center', va='center',
                fontsize=14, color='black', transform=axs[0, 0].transAxes)
axs[0, 1].imshow(gray_img, cmap='gray')
axs[0, 1].set_title('Grayscale Image')
axs[0, 2].imshow(median_img, cmap='gray')
axs[0, 2].set_title('Median Filter')
axs[1, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[1, 0].set_title('Original Image')
axs[1, 1].imshow(min_img, cmap='gray')
axs[1, 1].set_title('Min Filter')
axs[1, 2].imshow(max_img, cmap='gray')
axs[1, 2].set_title('Max Filter')

for ax in axs.flat:
    ax.axis('off')

plt.show()
