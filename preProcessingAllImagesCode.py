import os
import cv2
import numpy as np
import pywt

# Specify the directory you want to loop through
directory_path_yes = './yes'
directory_path_no = './no'
directory_path_yes_preprocessed = './yesPreprocessed'
directory_path_no_preprocessed = './noPreprocessed'


def pre_processing(filepath):
    image = cv2.imread(filepath)
    coeffs = pywt.wavedec2(image, 'haar', level=1)
    if len(coeffs) > 2:
        coeffs = list(coeffs)
        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], 0.2 * np.max(coeffs[i]), mode='soft')
    denoised_image = pywt.waverec2(coeffs, 'haar')
    gray_image = cv2.cvtColor(denoised_image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(gray_image)
    threshold_value = 220
    _, binary_image = cv2.threshold(equalized_image, threshold_value, 255, cv2.THRESH_BINARY)
    return binary_image


# Loop through all files in the directory
for filename in os.listdir(directory_path_no):
    file_path = os.path.join(directory_path_no, filename)

    # Check if the path is a file (not a directory)
    if os.path.isfile(file_path):
        binary_image = pre_processing(file_path)
        output_path = os.path.join(directory_path_no_preprocessed, filename)
        cv2.imwrite(output_path, binary_image)
