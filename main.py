import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt

# Load the image
imageyes = cv2.imread('Y27.jpg')
imageno = cv2.imread('noimg.jpg')

# Perform wavelet transform
coeffs1 = pywt.wavedec2(imageyes, 'haar', level=1)
coeffs2 = pywt.wavedec2(imageno, 'haar', level=1)

# Threshold the coefficients
if len(coeffs1) > 2:
    coeffs1 = list(coeffs1)
    for i in range(1, len(coeffs1)):
        coeffs1[i] = pywt.threshold(coeffs1[i], 0.2 * np.max(coeffs1[i]), mode='soft')
if len(coeffs2) > 2:
    coeffs2 = list(coeffs2)
    for i in range(1, len(coeffs2)):
        coeffs2[i] = pywt.threshold(coeffs2[i], 0.2 * np.max(coeffs2[i]), mode='soft')

# Perform inverse wavelet transform
denoised_image_yes = pywt.waverec2(coeffs1, 'haar')
denoised_image_no = pywt.waverec2(coeffs2, 'haar')

# Convert the image to grayscale
gray_image_yes = cv2.cvtColor(denoised_image_yes.astype(np.uint8), cv2.COLOR_BGR2GRAY)
gray_image_no = cv2.cvtColor(denoised_image_no.astype(np.uint8), cv2.COLOR_BGR2GRAY)

# Apply histogram equalization
equalized_image_yes = cv2.equalizeHist(gray_image_yes)
equalized_image_no = cv2.equalizeHist(gray_image_no)

#calculating histogram of both images
hist_yes = cv2.calcHist([equalized_image_yes], [0], None, [256], [0,256])
hist_no = cv2.calcHist([equalized_image_no], [0], None, [256], [0,256])

# Plot the histograms
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.title('Yes Image Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.bar(np.arange(256), hist_yes[:, 0], width=1)
plt.xlim([0, 256])
plt.grid()

plt.subplot(122)
plt.title('No image histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.bar(np.arange(256), hist_no[:, 0], width=1)
plt.xlim([0, 256])
plt.grid()

plt.tight_layout()
plt.show()

threshold_value = 175

# Apply binary thresholding
_, binary_image_yes = cv2.threshold(equalized_image_yes, threshold_value, 255, cv2.THRESH_BINARY)
_, binary_image_no = cv2.threshold(equalized_image_no, threshold_value, 255, cv2.THRESH_BINARY)


# Display the original and blurred images
cv2.imshow('Original Image with tumor', imageyes)
cv2.imshow('processed image of mri with tumor', binary_image_yes)
cv2.imshow('Original Image without tumor', imageno)
cv2.imshow('processed image of mri without tumor', binary_image_no)
cv2.waitKey(0)
cv2.destroyAllWindows()
